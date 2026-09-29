"""ModelPlacementController rebind operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping, Sequence

from ..model_placement_contracts.common import (
    REQUEST_HELPER_REBIND_STATES,
    ModelPlacementControllerError,
    _text,
    _integer,
    _sha256,
    _identities,
)
from ..model_placement_contracts.requests import (
    RequestHelperEnvelopeBinding,
    RequestHelperAttachment,
    RequestHelperRebind,
    RequestPlacementBinding,
)


def request_helper_rebind(
    controller,
    request_id: str,
    target_generation: int,
    *,
    observed_at_us: int,
) -> RequestHelperRebind:
    """Request an acknowledged helper quiescence for one layout change."""

    request_id = _text("request helper rebind request", request_id)
    target_generation = _integer(
        "request helper rebind target generation",
        target_generation,
        1,
    )
    _integer("request helper rebind time", observed_at_us)
    binding = controller._request_bindings.get(request_id)
    attachment = None if binding is None else binding.helper_attachment
    envelope = None if binding is None else binding.helper_envelope
    previous = controller._request_helper_rebinds.get(request_id)
    zero_quiesced = (
        attachment is not None
        and attachment.fraction_ppm == 0
        and binding.fraction_ppm == 0
        and not attachment.lease_tokens
        and attachment.lease_reserved_until_us is None
    )
    if (
        binding is None
        or request_id not in controller._acquired_request_ids
        or attachment is None
        or envelope is None
    ):
        raise ModelPlacementControllerError(
            "request helper rebind requires an active acquired helper"
        )
    if previous is None and not zero_quiesced and (
        attachment.fraction_ppm <= 0 or binding.fraction_ppm <= 0
    ):
        raise ModelPlacementControllerError(
            "request helper rebind requires an active acquired helper"
        )
    source = controller._phone_layouts.get(
        attachment.phone_layout_generation
    )
    target = controller._phone_layouts.get(target_generation)
    if previous is not None and previous.target_generation == target_generation:
        if (
            target is None
            or target.state not in {"PROPOSED", "PREPARING"}
            or target.layout.geometry_sha256
                != previous.target_geometry_sha256
            or previous.source_generation
                != attachment.phone_layout_generation
        ):
            raise ModelPlacementControllerError(
                "request helper rebind layout identity is invalid"
            )
        return previous
    if previous is not None and previous.target_generation != target_generation:
        stale_target = controller._phone_layouts.get(previous.target_generation)
        if (
            stale_target is not None
            and stale_target.state in {"PROPOSED", "PREPARING"}
        ):
            raise ModelPlacementControllerError(
                "request helper rebind is already active"
            )
        # The earlier target was rolled back; the retry resumes the
        # same quiescence against the replacement proposal.
        controller._request_helper_rebinds.pop(request_id, None)
        controller._record_request_helper_event(
            request_id,
            "REBIND_RETARGETED",
            observed_at_us,
            {
                **previous.to_json(),
                "new_target_generation": target_generation,
            },
        )
    if (
        source is None
        or source.state not in {"READY", "DRAINING"}
        or source.layout.geometry_sha256
            != attachment.phone_layout_geometry_sha256
        or target is None
        or target.state not in {"PROPOSED", "PREPARING"}
        or target.generation == source.generation
    ):
        raise ModelPlacementControllerError(
            "request helper rebind layout identity is invalid"
        )
    allowed_identities = tuple(
        row for row in attachment.phone_session_identities
        if row.session_id in set(attachment.allowed_session_ids)
    )
    if attachment.fraction_ppm > 0 and (
        not allowed_identities
        or not controller._session_identities_are_ready(allowed_identities)
    ):
        raise ModelPlacementControllerError(
            "request helper rebind session identity is not ready"
        )
    if (
        envelope.phone_layout_generation != source.generation
        or envelope.phone_layout_geometry_sha256
            != source.layout.geometry_sha256
        or envelope.desktop_parent_route_id != binding.base.route_id
        or envelope.desktop_placement_sha256
            != binding.base.desktop_placement_sha256
    ):
        raise ModelPlacementControllerError(
            "request helper rebind changed the desktop base"
        )
    source_sessions = (
        set(attachment.allowed_session_ids)
        if previous is None else
        set(previous.retained_session_ids)
            | set(previous.removed_session_ids)
    )
    retained = tuple(sorted(
        session_id for session_id in source_sessions
        if controller._sessions_unchanged(source, target, (session_id,))
    ))
    removed = tuple(sorted(source_sessions - set(retained)))
    if (
        not removed
        or not set(removed).issubset(
            set(target.layout.changed_session_ids)
        )
        or set(retained) | set(removed) != source_sessions
    ):
        raise ModelPlacementControllerError(
            "request helper rebind sessions differ from the transition"
        )
    source_shards = {
        row.session_id: row for row in source.layout.shards
    }
    retained_layer_mask = 0
    for session_id in retained:
        retained_layer_mask |= source_shards[session_id].layer_mask
    source_allowed = (
        attachment.allowed_session_ids
        if previous is None else previous.source_allowed_session_ids
    )
    masked_quiesced = bool(
        attachment.allowed_session_ids == retained
        and not set(attachment.allowed_session_ids) & set(removed)
        and retained
    )
    quiesced = masked_quiesced or zero_quiesced
    rebind = RequestHelperRebind(
        request_id=request_id,
        source_generation=source.generation,
        target_generation=target.generation,
        source_geometry_sha256=source.layout.geometry_sha256,
        target_geometry_sha256=target.layout.geometry_sha256,
        retained_session_ids=retained,
        removed_session_ids=removed,
        source_allowed_session_ids=source_allowed,
        target_allowed_session_ids=retained,
        retained_layer_mask=retained_layer_mask,
        previous_fraction_ppm=(
            attachment.fraction_ppm
            if previous is None
            else previous.previous_fraction_ppm
        ),
        state=(
            "QUIESCED"
            if quiesced and (
                previous is None or previous.state == "QUIESCED"
            )
            else "REQUESTED"
        ),
        drain_policy_sha256=(
            None if previous is None else previous.drain_policy_sha256
        ),
    )
    current = controller._request_helper_rebinds.get(request_id)
    if current is not None:
        if replace(current, state=rebind.state) != rebind:
            raise ModelPlacementControllerError(
                "request helper rebind is already active"
            )
        return current
    controller._request_helper_rebinds[request_id] = rebind
    controller._record_request_helper_event(
        request_id,
        "REBIND_REQUESTED"
        if rebind.state == "REQUESTED"
        else "REBIND_RESUMED_QUIESCED",
        observed_at_us,
        rebind.to_json(),
    )
    return rebind


def bind_request_helper_rebind_drain_policy(
    controller,
    request_id: str,
    target_generation: int,
    policy_sha256: str,
    *,
    observed_at_us: int,
) -> RequestHelperRebind:
    """Bind one runtime mask control to an active session rebind."""

    request_id = _text("request helper rebind request", request_id)
    target_generation = _integer(
        "request helper rebind target generation", target_generation, 1
    )
    policy = _sha256(
        "request helper rebind drain policy", policy_sha256
    )
    _integer("request helper rebind time", observed_at_us)
    rebind = controller._request_helper_rebinds.get(request_id)
    if (
        rebind is None
        or rebind.target_generation != target_generation
        or not rebind.target_allowed_session_ids
    ):
        raise ModelPlacementControllerError(
            "request helper rebind cannot bind a drain policy"
        )
    if rebind.drain_policy_sha256 is not None:
        if rebind.drain_policy_sha256 != policy:
            raise ModelPlacementControllerError(
                "request helper rebind drain policy changed"
            )
        return rebind
    if rebind.state != "REQUESTED":
        raise ModelPlacementControllerError(
            "request helper rebind cannot bind a drain policy"
        )
    updated = replace(rebind, drain_policy_sha256=policy)
    controller._request_helper_rebinds[request_id] = updated
    controller._record_request_helper_event(
        request_id,
        "REBIND_DRAIN_POLICY_BOUND",
        observed_at_us,
        updated.to_json(),
    )
    return updated


def mark_request_helper_rebind_quiesced(
    controller,
    request_id: str,
    target_generation: int,
    *,
    observed_at_us: int,
    allowed_session_ids: Sequence[str] | None = None,
    drain_policy_sha256: str | None = None,
) -> RequestHelperRebind:
    """Record an acknowledged session mask at a decode boundary."""

    request_id = _text("request helper rebind request", request_id)
    target_generation = _integer(
        "request helper rebind target generation",
        target_generation,
        1,
    )
    _integer("request helper rebind time", observed_at_us)
    rebind = controller._request_helper_rebinds.get(request_id)
    binding = controller._request_bindings.get(request_id)
    attachment = None if binding is None else binding.helper_attachment
    target = controller._phone_layouts.get(target_generation)
    if (
        rebind is not None
        and binding is not None
        and attachment is not None
    ):
        acknowledged = (
            attachment.allowed_session_ids
            if allowed_session_ids is None else
            _identities(
                "request helper acknowledged session",
                allowed_session_ids,
            )
        )
        explicit_mask = allowed_session_ids is not None
        if explicit_mask and (
            acknowledged != rebind.target_allowed_session_ids
            or rebind.drain_policy_sha256 is None
        ):
            raise ModelPlacementControllerError(
                "request helper rebind is not safely quiesced"
            )
        if rebind.drain_policy_sha256 is not None:
            if (
                not explicit_mask
                or drain_policy_sha256 != rebind.drain_policy_sha256
            ):
                raise ModelPlacementControllerError(
                    "request helper drain acknowledgement differs"
                )
        if explicit_mask:
            attachment = replace(
                attachment, allowed_session_ids=acknowledged
            )
            binding = replace(binding, helper_attachment=attachment)
            controller._request_bindings[request_id] = binding
    masked = bool(
        rebind is not None
        and attachment is not None
        and attachment.allowed_session_ids
            == rebind.target_allowed_session_ids
        and not (
            set(attachment.allowed_session_ids)
            & set(rebind.removed_session_ids)
        )
    )
    active_masked = bool(
        masked
        and rebind is not None
        and rebind.target_allowed_session_ids
        and attachment is not None
        and binding is not None
        and attachment.fraction_ppm == rebind.previous_fraction_ppm
        and binding.fraction_ppm == rebind.previous_fraction_ppm
        and (
            rebind.previous_fraction_ppm == 0
            or bool(attachment.lease_tokens)
            and attachment.lease_reserved_until_us is not None
        )
    )
    fully_quiesced = bool(
        attachment is not None
        and binding is not None
        and attachment.fraction_ppm == 0
        and binding.fraction_ppm == 0
        and not attachment.lease_tokens
        and attachment.lease_reserved_until_us is None
    )
    if (
        rebind is None
        or rebind.target_generation != target_generation
        or rebind.state not in REQUEST_HELPER_REBIND_STATES
        or binding is None
        or request_id not in controller._acquired_request_ids
        or attachment is None
        or attachment.phone_layout_generation
            != rebind.source_generation
        or attachment.phone_layout_geometry_sha256
            != rebind.source_geometry_sha256
        or not (active_masked or fully_quiesced)
        or target is None
        or target.state not in {"PROPOSED", "PREPARING"}
        or target.layout.geometry_sha256
            != rebind.target_geometry_sha256
    ):
        raise ModelPlacementControllerError(
            "request helper rebind is not safely quiesced"
        )
    if rebind.state == "QUIESCED":
        return rebind
    quiesced = replace(rebind, state="QUIESCED")
    controller._request_helper_rebinds[request_id] = quiesced
    controller._sync_phone_session_references()
    controller._record_request_helper_event(
        request_id,
        "REBIND_QUIESCED",
        observed_at_us,
        quiesced.to_json(),
    )
    return quiesced


def cancel_request_helper_rebind(
    controller,
    request_id: str,
    *,
    observed_at_us: int,
    reason: str,
) -> None:
    request_id = _text("request helper rebind request", request_id)
    _integer("request helper rebind time", observed_at_us)
    rebind = controller._request_helper_rebinds.pop(request_id, None)
    if rebind is None:
        return
    controller._record_request_helper_event(
        request_id,
        "REBIND_CANCELLED",
        observed_at_us,
        {**rebind.to_json(), "reason": _text(
            "request helper rebind cancellation reason", reason
        )},
    )


def request_helper_rebind_state(
    controller, request_id: str
) -> Mapping[str, object] | None:
    request_id = _text("request helper rebind request", request_id)
    rebind = controller._request_helper_rebinds.get(request_id)
    return (
        None if rebind is None
        else MappingProxyType(rebind.to_json())
    )


def commit_request_helper_rebind(
    controller,
    request_id: str,
    envelope: RequestHelperEnvelopeBinding,
    *,
    resident_component_identity_sha256: str,
    start_token_index: int,
    observed_at_us: int,
) -> RequestPlacementBinding:
    """Replace only a quiesced helper identity after its target is ready."""

    request_id = _text("request helper rebind request", request_id)
    if not isinstance(envelope, RequestHelperEnvelopeBinding):
        raise ModelPlacementControllerError(
            "request helper rebound envelope is invalid"
        )
    envelope = controller._normalize_helper_envelope(envelope)
    component = _sha256(
        "request helper rebound component",
        resident_component_identity_sha256,
    )
    token_index = _integer(
        "request helper rebound start token", start_token_index
    )
    _integer("request helper rebound time", observed_at_us)
    rebind = controller._request_helper_rebinds.get(request_id)
    binding = controller._request_bindings.get(request_id)
    old_attachment = (
        None if binding is None else binding.helper_attachment
    )
    old_envelope = None if binding is None else binding.helper_envelope
    target = (
        None if rebind is None else
        controller._phone_layouts.get(rebind.target_generation)
    )
    ready = controller.ready_phone_layout()
    masked_active = bool(
        rebind is not None
        and old_attachment is not None
        and binding is not None
        and old_attachment.allowed_session_ids
            == rebind.target_allowed_session_ids
        and not (
            set(old_attachment.allowed_session_ids)
            & set(rebind.removed_session_ids)
        )
        and old_attachment.fraction_ppm
            == rebind.previous_fraction_ppm
        and binding.fraction_ppm == rebind.previous_fraction_ppm
        and (
            rebind.previous_fraction_ppm == 0
            or bool(old_attachment.lease_tokens)
            and old_attachment.lease_reserved_until_us is not None
        )
    )
    fully_quiesced = bool(
        old_attachment is not None
        and binding is not None
        and old_attachment.fraction_ppm == 0
        and binding.fraction_ppm == 0
        and not old_attachment.lease_tokens
        and old_attachment.lease_reserved_until_us is None
    )
    if (
        rebind is None
        or rebind.state != "QUIESCED"
        or binding is None
        or request_id not in controller._acquired_request_ids
        or old_attachment is None
        or old_envelope is None
        or old_attachment.phone_layout_generation
            != rebind.source_generation
        or old_attachment.phone_layout_geometry_sha256
            != rebind.source_geometry_sha256
        or not (masked_active or fully_quiesced)
        or target is None
        or ready is None
        or target.generation != ready.generation
        or target.state != "READY"
        or target.layout.geometry_sha256
            != rebind.target_geometry_sha256
        or token_index < old_attachment.start_token_index
        or not controller._session_identities_are_ready(
            envelope.phone_session_identities
        )
    ):
        raise ModelPlacementControllerError(
            "request helper rebind is not ready to commit"
        )
    target_shards = tuple(
        row for row in target.layout.shards
        if row.artifact_sha256 == binding.base.artifact_sha256
    )
    target_sessions = tuple(sorted(
        row.session_id for row in target_shards
    ))
    target_mask = 0
    for shard in target_shards:
        target_mask |= shard.layer_mask
    if (
        not target_shards
        or envelope.phone_layout_generation != target.generation
        or envelope.phone_layout_geometry_sha256
            != target.layout.geometry_sha256
        or envelope.desktop_parent_route_id != binding.base.route_id
        or envelope.desktop_placement_sha256
            != binding.base.desktop_placement_sha256
        or envelope.phone_session_ids != target_sessions
        or envelope.assisted_layer_mask != target_mask
        or envelope.maximum_columns > min(
            row.maximum_columns for row in target_shards
        )
        or (
            masked_active
            and rebind.previous_fraction_ppm > 0
            and rebind.previous_fraction_ppm
                not in envelope.allowed_fractions_ppm
        )
        or (
            bool(rebind.retained_session_ids)
            and not controller._sessions_unchanged(
                controller.phone_layout(rebind.source_generation),
                target,
                rebind.retained_session_ids,
            )
        )
    ):
        raise ModelPlacementControllerError(
            "request helper rebound envelope differs from the target"
        )
    committed_fraction_ppm = (
        rebind.previous_fraction_ppm if masked_active else 0
    )
    new_attachment = RequestHelperAttachment(
        phone_layout_generation=target.generation,
        phone_layout_geometry_sha256=target.layout.geometry_sha256,
        resident_component_identity_sha256=component,
        operator_plan_sha256=envelope.operator_plan_sha256,
        phone_session_ids=envelope.phone_session_ids,
        allowed_session_ids=(
            rebind.target_allowed_session_ids
            if masked_active else envelope.phone_session_ids
        ),
        phone_session_identities=(
            envelope.phone_session_identities
        ),
        start_token_index=token_index,
        fraction_ppm=committed_fraction_ppm,
        lease_tokens=(
            old_attachment.lease_tokens if masked_active else ()
        ),
        lease_reserved_until_us=(
            old_attachment.lease_reserved_until_us
            if masked_active else None
        ),
        completed_phone_calls=old_attachment.completed_phone_calls,
    )
    result = replace(
        binding,
        helper_envelope=envelope,
        helper_attachment=new_attachment,
        fraction_ppm=committed_fraction_ppm,
    )
    controller._request_bindings[request_id] = result
    controller._request_helper_rebinds.pop(request_id, None)
    controller._sync_phone_session_references()
    controller._record_request_helper_event(
        request_id,
        "HELPER_REBOUND",
        observed_at_us,
        {
            **rebind.to_json(),
            "new_operator_plan_sha256": envelope.operator_plan_sha256,
            "old_operator_plan_sha256": (
                old_envelope.operator_plan_sha256
            ),
            "result": result.to_json(),
        },
    )
    controller._collect_drained_phone_layouts()
    return result
