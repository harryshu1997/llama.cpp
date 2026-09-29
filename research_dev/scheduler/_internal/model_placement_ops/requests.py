"""ModelPlacementController requests operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..model_placement_contracts.common import (
    ModelPlacementControllerError,
    _text,
    _integer,
    _sha256,
    _identities,
)
from ..model_placement_contracts.requests import (
    RequestBasePlacementBinding,
    RequestHelperEnvelopeBinding,
    RequestHelperAttachment,
    RequestPlacementBinding,
)


def bind_dispatched_request(
    controller,
    request_id: str,
    artifact_sha256: str,
    route_id: str,
    resident_component_identity_sha256: str,
    fraction_ppm: int,
    *,
    desktop_placement_sha256: str | None = None,
    kv_cache_owner_id: str | None = None,
    sequence_identity: str | None = None,
    server_slot_id: int | None = None,
    helper_envelope: RequestHelperEnvelopeBinding | None = None,
    output_tokens: int | None = None,
) -> None:
    request_id = _text("model placement request", request_id)
    artifact = _sha256(
        "model placement request artifact", artifact_sha256
    )
    route = _text("model placement request route", route_id)
    component = _sha256(
            "model placement request resident component",
            resident_component_identity_sha256,
    )
    desktop_placement = (
        component
        if desktop_placement_sha256 is None
        else _sha256(
            "model placement request desktop placement",
            desktop_placement_sha256,
        )
    )
    if helper_envelope is not None:
        if not isinstance(
            helper_envelope, RequestHelperEnvelopeBinding
        ):
            raise ModelPlacementControllerError(
                "request helper envelope is invalid"
            )
        helper_envelope = controller._normalize_helper_envelope(
            helper_envelope
        )
    binding = RequestPlacementBinding(
        base=RequestBasePlacementBinding(
            artifact_sha256=artifact,
            route_id=route,
            desktop_placement_sha256=desktop_placement,
            resident_component_identity_sha256=component,
            kv_cache_owner_id=(
                request_id
                if kv_cache_owner_id is None
                else _text(
                    "model placement request KV owner",
                    kv_cache_owner_id,
                )
            ),
            sequence_identity=(
                request_id
                if sequence_identity is None
                else _text(
                    "model placement request sequence identity",
                    sequence_identity,
                )
            ),
            server_slot_id=server_slot_id,
        ),
        helper_envelope=helper_envelope,
        helper_attachment=None,
        fraction_ppm=fraction_ppm,
    )
    _integer("model placement request fraction", fraction_ppm)
    if fraction_ppm > 1_000_000:
        raise ModelPlacementControllerError(
            "model placement request fraction is invalid"
        )
    previous = controller._request_bindings.get(request_id)
    if (
        request_id in controller._acquired_request_ids
        and previous is not None
        and previous != binding
    ):
        raise ModelPlacementControllerError(
            "dispatched request placement is immutable"
        )
    if output_tokens is not None:
        output_tokens = _integer(
            "model placement request output tokens",
            output_tokens,
            1,
        )
        progress = controller._request_decode_progress.get(request_id)
        if progress is not None and progress[0] != output_tokens:
            raise ModelPlacementControllerError(
                "request decode work identity differs"
            )
        controller._request_decode_progress[request_id] = (
            output_tokens,
            0 if progress is None else progress[1],
        )
    controller._request_bindings[request_id] = binding


def mark_request_acquired(controller, request_id: str) -> None:
    request_id = _text("model placement request", request_id)
    if request_id not in controller._request_bindings:
        raise ModelPlacementControllerError(
            "acquired request lacks a placement binding"
        )
    controller._acquired_request_ids.add(request_id)
    controller._note_helper_state_change()


def request_is_acquired(controller, request_id: str) -> bool:
    request_id = _text("model placement request", request_id)
    return request_id in controller._acquired_request_ids


def record_request_decode_progress(
    controller, request_id: str, token_index: int
) -> int:
    request_id = _text("model placement request", request_id)
    progress = controller._request_decode_progress.get(request_id)
    if (
        request_id not in controller._acquired_request_ids
        or progress is None
    ):
        raise ModelPlacementControllerError(
            "decode progress requires an acquired request"
        )
    output_tokens, current = progress
    token = _integer("request decode token", token_index)
    if token > output_tokens:
        raise ModelPlacementControllerError(
            "request decode progress exceeds requested work"
        )
    if token > current:
        controller._request_decode_progress[request_id] = (
            output_tokens, token
        )
        return token
    return current


def remaining_request_decode_tokens(
    controller, request_id: str, output_tokens: int
) -> int:
    request_id = _text("model placement request", request_id)
    expected = _integer(
        "model placement request output tokens", output_tokens, 1
    )
    progress = controller._request_decode_progress.get(request_id)
    if progress is None:
        return expected
    if progress[0] != expected:
        raise ModelPlacementControllerError(
            "request decode work identity differs"
        )
    return max(0, expected - progress[1])


def request_decode_token_index(controller, request_id: str) -> int:
    request_id = _text("model placement request", request_id)
    progress = controller._request_decode_progress.get(request_id)
    if progress is None:
        raise ModelPlacementControllerError(
            "request decode progress is absent"
        )
    return progress[1]


def bind_request_server_slot(
    controller,
    request_id: str,
    *,
    sequence_identity: str,
    server_slot_id: int,
) -> None:
    request_id = _text("model placement request", request_id)
    binding = controller._request_bindings.get(request_id)
    if binding is None or request_id not in controller._acquired_request_ids:
        raise ModelPlacementControllerError(
            "server slot requires an acquired request"
        )
    if binding.base.sequence_identity != _text(
        "model placement request sequence identity",
        sequence_identity,
    ):
        raise ModelPlacementControllerError(
            "server slot sequence identity differs"
        )
    slot = _integer("model placement request server slot", server_slot_id)
    current = binding.base.server_slot_id
    if current is not None and current != slot:
        raise ModelPlacementControllerError(
            "acquired request server slot is immutable"
        )
    if current is None:
        controller._request_bindings[request_id] = replace(
            binding,
            base=replace(binding.base, server_slot_id=slot),
        )


def bind_request_helper_envelope(
    controller,
    request_id: str,
    envelope: RequestHelperEnvelopeBinding,
    *,
    observed_at_us: int,
) -> RequestPlacementBinding:
    """Bind one exact helper generation without changing the base route."""

    request_id = _text("model placement request", request_id)
    if not isinstance(envelope, RequestHelperEnvelopeBinding):
        raise ModelPlacementControllerError(
            "request helper envelope is invalid"
        )
    envelope = controller._normalize_helper_envelope(envelope)
    binding = controller._request_bindings.get(request_id)
    if binding is None or request_id not in controller._acquired_request_ids:
        raise ModelPlacementControllerError(
            "helper envelope requires an acquired request"
        )
    layout = controller.phone_layout(envelope.phone_layout_generation)
    if (
        layout.layout.geometry_sha256
            != envelope.phone_layout_geometry_sha256
        or not layout.covers_artifact(binding.base.artifact_sha256)
        or envelope.desktop_parent_route_id != binding.base.route_id
        or envelope.desktop_placement_sha256
            != binding.base.desktop_placement_sha256
        or not controller._session_identities_are_ready(
            envelope.phone_session_identities
        )
    ):
        raise ModelPlacementControllerError(
            "request helper envelope differs from its base placement"
        )
    previous = binding.helper_envelope
    if previous is not None:
        if previous == envelope:
            return binding
        attachment = binding.helper_attachment
        if (
            attachment is not None
            and (
                attachment.completed_phone_calls > 0
                or attachment.fraction_ppm != 0
                or binding.fraction_ppm != 0
                or bool(attachment.lease_tokens)
                or attachment.lease_reserved_until_us is not None
            )
        ):
            raise ModelPlacementControllerError(
                "request helper envelope identity is immutable"
            )
        result = replace(
            binding,
            helper_envelope=envelope,
            helper_attachment=None,
            fraction_ppm=0,
        )
        controller._request_bindings[request_id] = result
        controller._record_request_helper_event(
            request_id,
            "INACTIVE_ENVELOPE_REPLACED",
            observed_at_us,
            {
                "new_envelope": envelope.to_json(),
                "old_envelope": previous.to_json(),
                "result": result.to_json(),
            },
        )
        return result
    result = replace(binding, helper_envelope=envelope)
    controller._request_bindings[request_id] = result
    controller._record_request_helper_event(
        request_id,
        "ENVELOPE_BOUND",
        observed_at_us,
        result.to_json(),
    )
    return result


def request_helper_envelope_is_additive(
    controller,
    request_id: str,
    envelope: RequestHelperEnvelopeBinding,
) -> bool:
    """Return whether a READY envelope only adds verified sessions."""

    try:
        request_id = _text("model placement request", request_id)
        if not isinstance(envelope, RequestHelperEnvelopeBinding):
            return False
        envelope = controller._normalize_helper_envelope(envelope)
        binding = controller._request_bindings.get(request_id)
        attachment = (
            None if binding is None else binding.helper_attachment
        )
        previous = None if binding is None else binding.helper_envelope
        target = controller.phone_layout(envelope.phone_layout_generation)
    except ModelPlacementControllerError:
        return False
    if (
        binding is None
        or request_id not in controller._acquired_request_ids
        or attachment is None
        or previous is None
        or request_id in controller._request_helper_rebinds
        or target.state != "READY"
        or attachment.fallback_outcome is not None
        or envelope.desktop_parent_route_id != binding.base.route_id
        or envelope.desktop_placement_sha256
            != binding.base.desktop_placement_sha256
        or envelope.activation_dtype != previous.activation_dtype
        or envelope.maximum_columns != previous.maximum_columns
        or envelope.assisted_layer_mask
            & previous.assisted_layer_mask
            != previous.assisted_layer_mask
        or not set(previous.allowed_fractions_ppm).issubset(
            envelope.allowed_fractions_ppm
        )
        or attachment.fraction_ppm
            not in envelope.allowed_fractions_ppm
        or envelope.resource_ids != previous.resource_ids
        or attachment.phone_session_ids != previous.phone_session_ids
        or attachment.phone_session_identities
            != previous.phone_session_identities
        or set(attachment.allowed_session_ids)
            != set(previous.phone_session_ids)
    ):
        return False
    previous_by_session = {
        row.session_id: row
        for row in previous.phone_session_identities
    }
    target_by_session = {
        row.session_id: row
        for row in envelope.phone_session_identities
    }
    if (
        not set(previous_by_session) < set(target_by_session)
        or any(
            target_by_session[session_id] != identity
            for session_id, identity in previous_by_session.items()
        )
        or not controller._session_identities_are_ready(
            envelope.phone_session_identities
        )
    ):
        return False
    target_shards = tuple(
        row for row in target.layout.shards
        if row.artifact_sha256 == binding.base.artifact_sha256
    )
    target_mask = 0
    for shard in target_shards:
        target_mask |= shard.layer_mask
    # A static co-helper (the second phone of a two-phone arm) serves layers
    # outside this layout; every envelope of the request carries them, so the
    # addition must keep exactly the previous envelope's co-helper layers and
    # add only this layout's shards.  Without a co-helper this is 0 and the
    # check is the exact-layout equality it always was.
    co_helper_mask = previous.assisted_layer_mask & ~target_mask
    return bool(
        target_shards
        and envelope.phone_session_ids == tuple(sorted(
            row.session_id for row in target_shards
        ))
        and envelope.assisted_layer_mask == target_mask | co_helper_mask
    )


def expand_request_helper_envelope(
    controller,
    request_id: str,
    envelope: RequestHelperEnvelopeBinding,
    *,
    resident_component_identity_sha256: str,
    start_token_index: int,
    fraction_ppm: int,
    lease_tokens: Sequence[str],
    lease_reserved_until_us: int | None,
    observed_at_us: int,
) -> RequestPlacementBinding:
    """Attach newly READY sessions at one decode boundary."""

    request_id = _text("model placement request", request_id)
    if not controller.request_helper_envelope_is_additive(
        request_id, envelope
    ):
        raise ModelPlacementControllerError(
            "request helper envelope addition is invalid"
        )
    envelope = controller._normalize_helper_envelope(envelope)
    binding = controller._request_bindings[request_id]
    previous = binding.helper_attachment
    assert previous is not None
    component = _sha256(
        "request helper expanded component",
        resident_component_identity_sha256,
    )
    token_index = _integer(
        "request helper expanded start token", start_token_index
    )
    fraction = _integer(
        "request helper expanded fraction", fraction_ppm
    )
    tokens = _identities(
        "request helper expanded lease token", lease_tokens
    )
    if (
        token_index < previous.start_token_index
        or fraction not in envelope.allowed_fractions_ppm
        or (fraction > 0) != bool(tokens)
        or (lease_reserved_until_us is None) != (not tokens)
    ):
        raise ModelPlacementControllerError(
            "request helper envelope addition is invalid"
        )
    expanded = RequestHelperAttachment(
        phone_layout_generation=envelope.phone_layout_generation,
        phone_layout_geometry_sha256=(
            envelope.phone_layout_geometry_sha256
        ),
        resident_component_identity_sha256=component,
        operator_plan_sha256=envelope.operator_plan_sha256,
        phone_session_ids=envelope.phone_session_ids,
        allowed_session_ids=envelope.phone_session_ids,
        phone_session_identities=envelope.phone_session_identities,
        start_token_index=token_index,
        fraction_ppm=fraction,
        lease_tokens=tokens,
        lease_reserved_until_us=lease_reserved_until_us,
        completed_phone_calls=previous.completed_phone_calls,
    )
    result = replace(
        binding,
        helper_envelope=envelope,
        helper_attachment=expanded,
        fraction_ppm=fraction,
    )
    controller._request_bindings[request_id] = result
    controller._sync_phone_session_references()
    controller._record_request_helper_event(
        request_id,
        "HELPER_EXPANDED",
        observed_at_us,
        {
            "new_envelope": envelope.to_json(),
            "old_envelope": binding.helper_envelope.to_json(),
            "result": result.to_json(),
        },
    )
    return result
