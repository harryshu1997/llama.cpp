"""ModelPlacementController attachment operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping, Sequence

from ..types import canonical_sha256
from ..model_placement_contracts.common import (
    ModelPlacementControllerError,
    _text,
    _integer,
    _sha256,
    _identities,
)
from ..model_placement_contracts.requests import RequestHelperAttachment, RequestPlacementBinding


def _record_request_helper_event(
    controller,
    request_id: str,
    kind: str,
    observed_at_us: int,
    values: Mapping[str, object],
) -> None:
    events = controller._request_helper_events
    body = {
        "event_index": (
            int(events[-1]["event_index"]) + 1 if events else 0
        ),
        "kind": _text("request helper event kind", kind),
        "observed_at_us": _integer(
            "request helper event time", observed_at_us
        ),
        "request_id": _text("request helper event request", request_id),
        "schema": "research-scheduler-request-helper-event-v1",
        **dict(values),
    }
    previous = events[-1] if events else None
    if (
        kind == "LEASES_RENEWED"
        and previous is not None
        and previous["kind"] == "LEASES_RENEWED"
        and previous["request_id"] == body["request_id"]
    ):
        # Consecutive renewals of one request collapse into the latest lease
        # state; the count and first time keep the renewal history auditable.
        body["renewal_count"] = int(previous.get("renewal_count", 1)) + 1
        body["first_observed_at_us"] = int(
            previous.get("first_observed_at_us", previous["observed_at_us"])
        )
        body["first_event_index"] = int(
            previous.get("first_event_index", previous["event_index"])
        )
        events.pop()
    controller._note_helper_state_change()
    events.append(MappingProxyType({
        **body,
        "event_sha256": canonical_sha256(body),
    }))
    excess = len(events) - controller.policy.maximum_events
    if excess > 0:
        # Evict renewals first so lifecycle and reason events survive the cap.
        renewals = [
            index for index, row in enumerate(events)
            if row["kind"] == "LEASES_RENEWED"
        ][:excess]
        for index in reversed(renewals):
            del events[index]
        excess -= len(renewals)
        if excess > 0:
            del events[:excess]


def attach_request_helper(
    controller,
    request_id: str,
    *,
    phone_layout_generation: int,
    phone_layout_geometry_sha256: str,
    resident_component_identity_sha256: str,
    operator_plan_sha256: str,
    start_token_index: int,
    fraction_ppm: int,
    lease_tokens: Sequence[str],
    lease_reserved_until_us: int | None,
    observed_at_us: int,
) -> RequestPlacementBinding:
    request_id = _text("model placement request", request_id)
    binding = controller._request_bindings.get(request_id)
    if binding is None or request_id not in controller._acquired_request_ids:
        raise ModelPlacementControllerError(
            "helper attachment requires an acquired request"
        )
    envelope = binding.helper_envelope
    if envelope is None:
        raise ModelPlacementControllerError(
            "request has no compatible helper envelope"
        )
    layout = controller.phone_layout(phone_layout_generation)
    if not layout.covers_artifact(binding.base.artifact_sha256):
        raise ModelPlacementControllerError(
            "phone layout does not cover the request artifact"
        )
    existing_attachment = binding.helper_attachment
    allowed_session_ids = (
        envelope.phone_session_ids
        if existing_attachment is None else
        existing_attachment.allowed_session_ids
    )
    allowed_session_id_set = set(allowed_session_ids)
    allowed_identities = tuple(
        identity for identity in envelope.phone_session_identities
        if identity.session_id in allowed_session_id_set
    )
    if (
        envelope.phone_layout_generation
            != phone_layout_generation
        or layout.layout.geometry_sha256
            != phone_layout_geometry_sha256
        or envelope.phone_layout_geometry_sha256
            != phone_layout_geometry_sha256
        or envelope.operator_plan_sha256 != operator_plan_sha256
        or envelope.desktop_parent_route_id != binding.base.route_id
        or envelope.desktop_placement_sha256
            != binding.base.desktop_placement_sha256
        or fraction_ppm not in envelope.allowed_fractions_ppm
        or not controller._session_identities_are_ready(
            allowed_identities
        )
    ):
        raise ModelPlacementControllerError(
            "request helper differs from its adaptive envelope"
        )
    attachment = RequestHelperAttachment(
        phone_layout_generation=phone_layout_generation,
        phone_layout_geometry_sha256=phone_layout_geometry_sha256,
        resident_component_identity_sha256=(
            resident_component_identity_sha256
        ),
        operator_plan_sha256=operator_plan_sha256,
        phone_session_ids=envelope.phone_session_ids,
        allowed_session_ids=allowed_session_ids,
        phone_session_identities=(
            envelope.phone_session_identities
        ),
        start_token_index=start_token_index,
        fraction_ppm=fraction_ppm,
        lease_tokens=tuple(lease_tokens),
        lease_reserved_until_us=lease_reserved_until_us,
        completed_phone_calls=(
            0
            if existing_attachment is None else
            existing_attachment.completed_phone_calls
        ),
    )
    previous = existing_attachment
    if previous is not None and (
        previous.phone_layout_generation
            != attachment.phone_layout_generation
        or previous.phone_layout_geometry_sha256
            != attachment.phone_layout_geometry_sha256
        or previous.operator_plan_sha256
            != attachment.operator_plan_sha256
        or previous.phone_session_identities
            != attachment.phone_session_identities
    ):
        raise ModelPlacementControllerError(
            "request helper attachment identity is immutable"
        )
    result = replace(
        binding,
        helper_attachment=attachment,
        fraction_ppm=fraction_ppm,
    )
    controller._request_bindings[request_id] = result
    controller._sync_phone_session_references()
    controller._record_request_helper_event(
        request_id,
        "ATTACHED" if previous is None else "FRACTION_CHANGED",
        observed_at_us,
        {
            **result.to_json(),
            "accepted": True,
            "selected_fraction_ppm": fraction_ppm,
            "session_generation_by_id": {
                row.session_id: row.session_generation
                for row in attachment.phone_session_identities
            },
            "template_layout_generation": phone_layout_generation,
        },
    )
    return result


def record_request_helper_work(
    controller,
    request_id: str,
    completed_phone_calls: int,
    *,
    observed_at_us: int,
) -> None:
    """Record physical work that makes an acquired layout a blocker."""

    request_id = _text("model placement request", request_id)
    calls = _integer(
        "request helper completed phone calls",
        completed_phone_calls,
        1,
    )
    binding = controller._request_bindings.get(request_id)
    attachment = None if binding is None else binding.helper_attachment
    if (
        binding is None
        or request_id not in controller._acquired_request_ids
        or attachment is None
        or binding.fraction_ppm <= 0
    ):
        raise ModelPlacementControllerError(
            "physical helper work lacks an active attachment"
        )
    updated = replace(
        attachment,
        completed_phone_calls=(
            attachment.completed_phone_calls + calls
        ),
    )
    result = replace(binding, helper_attachment=updated)
    controller._request_bindings[request_id] = result
    controller._sync_phone_session_references()
    controller._record_request_helper_event(
        request_id,
        "PHYSICAL_WORK_RECORDED",
        observed_at_us,
        {
            "completed_phone_calls": updated.completed_phone_calls,
            "phone_layout_generation": (
                updated.phone_layout_generation
            ),
        },
    )


def update_request_fraction(
    controller,
    request_id: str,
    fraction_ppm: int,
    *,
    resident_component_identity_sha256: str | None = None,
    observed_at_us: int = 0,
) -> None:
    request_id = _text("model placement request", request_id)
    binding = controller._request_bindings.get(request_id)
    if binding is None:
        raise ModelPlacementControllerError(
            "model placement request is not dispatched"
        )
    _integer("model placement request fraction", fraction_ppm)
    if fraction_ppm > 1_000_000:
        raise ModelPlacementControllerError(
            "model placement request fraction is invalid"
        )
    attachment = binding.helper_attachment
    if fraction_ppm > 0 and binding.helper_envelope is not None:
        if attachment is None:
            raise ModelPlacementControllerError(
                "adaptive fraction lacks a helper attachment"
            )
        if fraction_ppm not in (
            binding.helper_envelope.allowed_fractions_ppm
        ):
            raise ModelPlacementControllerError(
                "adaptive fraction exceeds the helper envelope"
            )
    if resident_component_identity_sha256 is not None:
        component = _sha256(
            "model placement request resident component",
            resident_component_identity_sha256,
        )
        expected = (
            binding.base.resident_component_identity_sha256
            if attachment is None
            else attachment.resident_component_identity_sha256
        )
        if expected != component:
            raise ModelPlacementControllerError(
                "adaptive fraction changed residency identity"
            )
    updated_attachment = attachment
    if attachment is not None:
        updated_attachment = replace(
            attachment,
            fraction_ppm=fraction_ppm,
            lease_tokens=(
                () if fraction_ppm == 0 else attachment.lease_tokens
            ),
            lease_reserved_until_us=(
                None
                if fraction_ppm == 0
                else attachment.lease_reserved_until_us
            ),
        )
    result = replace(
        binding,
        helper_attachment=updated_attachment,
        fraction_ppm=fraction_ppm,
    )
    controller._request_bindings[request_id] = result
    controller._sync_phone_session_references()
    controller._record_request_helper_event(
        request_id,
        "FRACTION_CHANGED",
        observed_at_us,
        result.to_json(),
    )


def renew_request_helper_leases(
    controller,
    request_id: str,
    *,
    lease_tokens: Sequence[str],
    reserved_until_us: int,
    observed_at_us: int,
) -> None:
    request_id = _text("model placement request", request_id)
    binding = controller._request_bindings.get(request_id)
    attachment = None if binding is None else binding.helper_attachment
    if (
        binding is None
        or request_id not in controller._acquired_request_ids
        or attachment is None
        or binding.fraction_ppm == 0
    ):
        raise ModelPlacementControllerError(
            "helper lease renewal requires an active attachment"
        )
    tokens = _identities("request helper lease token", lease_tokens)
    horizon = _integer(
        "request helper lease horizon", reserved_until_us, 1
    )
    if (
        tokens != attachment.lease_tokens
        or attachment.lease_reserved_until_us is None
        or horizon < attachment.lease_reserved_until_us
    ):
        raise ModelPlacementControllerError(
            "request helper lease renewal differs"
        )
    if horizon == attachment.lease_reserved_until_us:
        return
    updated = replace(
        attachment,
        lease_reserved_until_us=horizon,
    )
    result = replace(binding, helper_attachment=updated)
    controller._request_bindings[request_id] = result
    controller._sync_phone_session_references()
    controller._record_request_helper_event(
        request_id,
        "LEASES_RENEWED",
        observed_at_us,
        result.to_json(),
    )


def detach_request_helper(
    controller,
    request_id: str,
    *,
    fallback_outcome: str,
    observed_at_us: int,
) -> RequestPlacementBinding:
    request_id = _text("model placement request", request_id)
    binding = controller._request_bindings.get(request_id)
    if binding is None:
        raise ModelPlacementControllerError(
            "model placement request is not dispatched"
        )
    attachment = binding.helper_attachment
    if attachment is None:
        return binding
    result = replace(
        binding,
        helper_attachment=replace(
            attachment,
            allowed_session_ids=attachment.allowed_session_ids,
            fraction_ppm=0,
            lease_tokens=(),
            lease_reserved_until_us=None,
            fallback_outcome=_text(
                "request helper fallback outcome", fallback_outcome
            ),
        ),
        fraction_ppm=0,
    )
    controller._request_bindings[request_id] = result
    controller._sync_phone_session_references()
    controller._record_request_helper_event(
        request_id, "DETACHED", observed_at_us, result.to_json()
    )
    return result


def release_request(controller, request_id: str) -> None:
    request_id = _text("model placement request", request_id)
    controller._request_bindings.pop(request_id, None)
    controller._request_helper_rebinds.pop(request_id, None)
    controller._acquired_request_ids.discard(request_id)
    controller._request_decode_progress.pop(request_id, None)
    controller._sync_phone_session_references()
    controller._collect_drained_phone_layouts()


def request_binding(
    controller, request_id: str
) -> Mapping[str, object] | None:
    request_id = _text("model placement request", request_id)
    row = controller._request_bindings.get(request_id)
    if row is None:
        return None
    return MappingProxyType({
        **row.to_json(),
        "artifact_sha256": row.base.artifact_sha256,
        "resident_component_identity_sha256": (
            row.base.resident_component_identity_sha256
            if row.helper_attachment is None
            else row.helper_attachment
                .resident_component_identity_sha256
        ),
        "route_id": row.base.route_id,
    })


def request_helper_events(
    controller, request_id: str | None = None
) -> tuple[Mapping[str, object], ...]:
    if request_id is not None:
        request_id = _text("model placement request", request_id)
    return tuple(
        MappingProxyType(dict(event))
        for event in controller._request_helper_events
        if request_id is None or event["request_id"] == request_id
    )


def record_request_helper_event(
    controller,
    request_id: str,
    kind: str,
    observed_at_us: int,
    values: Mapping[str, object],
) -> None:
    controller._record_request_helper_event(
        request_id, kind, observed_at_us, values
    )
