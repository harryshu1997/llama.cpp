"""Model demand, layout and request-binding records: requests."""

from __future__ import annotations

from dataclasses import dataclass

from ..phone_shards import PhoneFfnSessionIdentity
from .common import (
    ModelPlacementControllerError,
    REQUEST_HELPER_REBIND_STATES,
    _identities,
    _integer,
    _sha256,
    _text,
)


@dataclass(frozen=True)
class RequestBasePlacementBinding:
    artifact_sha256: str
    route_id: str
    desktop_placement_sha256: str
    resident_component_identity_sha256: str
    kv_cache_owner_id: str
    sequence_identity: str
    server_slot_id: int | None = None

    def __post_init__(self) -> None:
        _sha256("request base artifact", self.artifact_sha256)
        _text("request base route", self.route_id)
        _sha256(
            "request base desktop placement",
            self.desktop_placement_sha256,
        )
        _sha256(
            "request base resident component",
            self.resident_component_identity_sha256,
        )
        _text("request base KV owner", self.kv_cache_owner_id)
        _text("request base sequence identity", self.sequence_identity)
        if self.server_slot_id is not None:
            _integer("request base server slot", self.server_slot_id)

    def to_json(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "desktop_placement_sha256": self.desktop_placement_sha256,
            "kv_cache_owner_id": self.kv_cache_owner_id,
            "resident_component_identity_sha256": (
                self.resident_component_identity_sha256
            ),
            "route_id": self.route_id,
            "sequence_identity": self.sequence_identity,
            "server_slot_id": self.server_slot_id,
        }


@dataclass(frozen=True)
class RequestHelperEnvelopeBinding:
    route_id: str
    operator_plan_sha256: str
    desktop_parent_route_id: str
    desktop_placement_sha256: str
    phone_layout_generation: int
    phone_layout_geometry_sha256: str
    activation_dtype: str
    assisted_layer_mask: int
    maximum_columns: int
    allowed_fractions_ppm: tuple[int, ...]
    phone_session_ids: tuple[str, ...]
    resource_ids: tuple[str, ...]
    phone_session_identities: tuple[PhoneFfnSessionIdentity, ...] = ()

    def __post_init__(self) -> None:
        _text("request helper route", self.route_id)
        _sha256("request helper operator plan", self.operator_plan_sha256)
        _text(
            "request helper desktop parent", self.desktop_parent_route_id
        )
        _sha256(
            "request helper desktop placement",
            self.desktop_placement_sha256,
        )
        _integer(
            "request helper phone layout generation",
            self.phone_layout_generation,
            1,
        )
        _sha256(
            "request helper phone layout geometry",
            self.phone_layout_geometry_sha256,
        )
        _text("request helper activation dtype", self.activation_dtype)
        mask = _integer(
            "request helper assisted layer mask",
            self.assisted_layer_mask,
            1,
        )
        if mask >= 1 << 64:
            raise ModelPlacementControllerError(
                "request helper layer mask exceeds 64 layers"
            )
        _integer("request helper maximum columns", self.maximum_columns, 1)
        fractions = tuple(sorted(
            _integer("request helper fraction", value)
            for value in self.allowed_fractions_ppm
        ))
        if (
            not fractions
            or fractions[0] != 0
            or not any(value > 0 for value in fractions)
            or fractions[-1] > 1_000_000
            or len(fractions) != len(set(fractions))
        ):
            raise ModelPlacementControllerError(
                "request helper fractions are invalid"
            )
        resources = _identities(
            "request helper resource", self.resource_ids
        )
        sessions = _identities(
            "request helper phone session", self.phone_session_ids
        )
        session_identities = tuple(sorted(
            self.phone_session_identities,
            key=lambda row: row.session_id,
        ))
        if (
            any(
                not isinstance(row, PhoneFfnSessionIdentity)
                for row in session_identities
            )
            or len({row.session_id for row in session_identities})
                != len(session_identities)
            or session_identities
                and tuple(row.session_id for row in session_identities)
                    != sessions
        ):
            raise ModelPlacementControllerError(
                "request helper phone session identities are invalid"
            )
        if not resources:
            raise ModelPlacementControllerError(
                "request helper resources are absent"
            )
        if not sessions:
            raise ModelPlacementControllerError(
                "request helper phone sessions are absent"
            )
        object.__setattr__(self, "allowed_fractions_ppm", fractions)
        object.__setattr__(self, "phone_session_ids", sessions)
        object.__setattr__(
            self, "phone_session_identities", session_identities
        )
        object.__setattr__(self, "resource_ids", resources)

    def to_json(self) -> dict[str, object]:
        return {
            "activation_dtype": self.activation_dtype,
            "allowed_fractions_ppm": list(
                self.allowed_fractions_ppm
            ),
            "assisted_layer_mask": self.assisted_layer_mask,
            "desktop_parent_route_id": self.desktop_parent_route_id,
            "desktop_placement_sha256": self.desktop_placement_sha256,
            "maximum_columns": self.maximum_columns,
            "operator_plan_sha256": self.operator_plan_sha256,
            "phone_layout_generation": self.phone_layout_generation,
            "phone_layout_geometry_sha256": (
                self.phone_layout_geometry_sha256
            ),
            "phone_session_ids": list(self.phone_session_ids),
            "phone_session_identities": [
                row.to_json() for row in self.phone_session_identities
            ],
            "resource_ids": list(self.resource_ids),
            "route_id": self.route_id,
        }


@dataclass(frozen=True)
class RequestHelperAttachment:
    phone_layout_generation: int
    phone_layout_geometry_sha256: str
    resident_component_identity_sha256: str
    operator_plan_sha256: str
    phone_session_ids: tuple[str, ...]
    allowed_session_ids: tuple[str, ...]
    start_token_index: int
    fraction_ppm: int
    lease_tokens: tuple[str, ...]
    lease_reserved_until_us: int | None
    phone_session_identities: tuple[PhoneFfnSessionIdentity, ...] = ()
    completed_phone_calls: int = 0
    fallback_outcome: str | None = None

    def __post_init__(self) -> None:
        _integer(
            "request helper layout generation",
            self.phone_layout_generation,
            1,
        )
        _sha256(
            "request helper attached geometry",
            self.phone_layout_geometry_sha256,
        )
        _sha256(
            "request helper attached component",
            self.resident_component_identity_sha256,
        )
        _sha256(
            "request helper attached operator plan",
            self.operator_plan_sha256,
        )
        sessions = _identities(
            "request helper attached phone session",
            self.phone_session_ids,
        )
        if not sessions:
            raise ModelPlacementControllerError(
                "request helper attached phone sessions are absent"
            )
        allowed_sessions = _identities(
            "request helper allowed phone session",
            self.allowed_session_ids,
        )
        if set(allowed_sessions) - set(sessions):
            raise ModelPlacementControllerError(
                "request helper allowed sessions exceed its binding"
            )
        session_identities = tuple(sorted(
            self.phone_session_identities,
            key=lambda row: row.session_id,
        ))
        if (
            any(
                not isinstance(row, PhoneFfnSessionIdentity)
                for row in session_identities
            )
            or len({row.session_id for row in session_identities})
                != len(session_identities)
            or session_identities
                and tuple(row.session_id for row in session_identities)
                    != sessions
        ):
            raise ModelPlacementControllerError(
                "request helper attached session identities are invalid"
            )
        _integer("request helper start token", self.start_token_index)
        fraction = _integer(
            "request helper attached fraction", self.fraction_ppm
        )
        if fraction > 1_000_000:
            raise ModelPlacementControllerError(
                "request helper attached fraction is invalid"
            )
        if fraction > 0 and not allowed_sessions:
            raise ModelPlacementControllerError(
                "active request helper has no allowed phone sessions"
            )
        tokens = _identities(
            "request helper lease token", self.lease_tokens
        )
        if fraction > 0 and not tokens:
            raise ModelPlacementControllerError(
                "active request helper lacks leases"
            )
        if self.lease_reserved_until_us is None:
            if tokens:
                raise ModelPlacementControllerError(
                    "request helper lease horizon is absent"
                )
        else:
            _integer(
                "request helper lease horizon",
                self.lease_reserved_until_us,
                1,
            )
            if not tokens:
                raise ModelPlacementControllerError(
                    "request helper lease horizon lacks leases"
                )
        if self.fallback_outcome is not None:
            _text(
                "request helper fallback outcome", self.fallback_outcome
            )
        _integer(
            "request helper completed phone calls",
            self.completed_phone_calls,
        )
        object.__setattr__(self, "lease_tokens", tokens)
        object.__setattr__(self, "phone_session_ids", sessions)
        object.__setattr__(
            self, "allowed_session_ids", allowed_sessions
        )
        object.__setattr__(
            self, "phone_session_identities", session_identities
        )

    def to_json(self) -> dict[str, object]:
        return {
            "fallback_outcome": self.fallback_outcome,
            "fraction_ppm": self.fraction_ppm,
            "lease_reserved_until_us": self.lease_reserved_until_us,
            "lease_tokens": list(self.lease_tokens),
            "completed_phone_calls": self.completed_phone_calls,
            "operator_plan_sha256": self.operator_plan_sha256,
            "allowed_session_ids": list(self.allowed_session_ids),
            "phone_layout_generation": self.phone_layout_generation,
            "phone_layout_geometry_sha256": (
                self.phone_layout_geometry_sha256
            ),
            "phone_session_ids": list(self.phone_session_ids),
            "phone_session_identities": [
                row.to_json() for row in self.phone_session_identities
            ],
            "resident_component_identity_sha256": (
                self.resident_component_identity_sha256
            ),
            "start_token_index": self.start_token_index,
        }


@dataclass(frozen=True)
class RequestHelperRebind:
    request_id: str
    source_generation: int
    target_generation: int
    source_geometry_sha256: str
    target_geometry_sha256: str
    retained_session_ids: tuple[str, ...]
    removed_session_ids: tuple[str, ...]
    source_allowed_session_ids: tuple[str, ...]
    target_allowed_session_ids: tuple[str, ...]
    retained_layer_mask: int
    previous_fraction_ppm: int
    state: str
    drain_policy_sha256: str | None = None

    def __post_init__(self) -> None:
        _text("request helper rebind request", self.request_id)
        source = _integer(
            "request helper rebind source generation",
            self.source_generation,
            1,
        )
        target = _integer(
            "request helper rebind target generation",
            self.target_generation,
            1,
        )
        if source == target:
            raise ModelPlacementControllerError(
                "request helper rebind generations are identical"
            )
        _sha256(
            "request helper rebind source geometry",
            self.source_geometry_sha256,
        )
        _sha256(
            "request helper rebind target geometry",
            self.target_geometry_sha256,
        )
        retained = _identities(
            "request helper rebind retained session",
            self.retained_session_ids,
        )
        removed = _identities(
            "request helper rebind removed session",
            self.removed_session_ids,
        )
        source_allowed = _identities(
            "request helper rebind source allowed session",
            self.source_allowed_session_ids,
        )
        target_allowed = _identities(
            "request helper rebind target allowed session",
            self.target_allowed_session_ids,
        )
        if not removed or set(retained) & set(removed):
            raise ModelPlacementControllerError(
                "request helper rebind session sets are invalid"
            )
        if (
            set(source_allowed) - (set(retained) | set(removed))
            or target_allowed != retained
        ):
            raise ModelPlacementControllerError(
                "request helper rebind allowed sessions are invalid"
            )
        layer_mask = _integer(
            "request helper rebind retained layer mask",
            self.retained_layer_mask,
        )
        if bool(retained) != bool(layer_mask):
            raise ModelPlacementControllerError(
                "request helper rebind retained layers are invalid"
            )
        fraction = _integer(
            "request helper rebind previous fraction",
            self.previous_fraction_ppm,
        )
        if fraction > 1_000_000:
            raise ModelPlacementControllerError(
                "request helper rebind previous fraction is invalid"
            )
        if self.state not in REQUEST_HELPER_REBIND_STATES:
            raise ModelPlacementControllerError(
                "request helper rebind state is invalid"
            )
        if self.drain_policy_sha256 is not None:
            _sha256(
                "request helper rebind drain policy",
                self.drain_policy_sha256,
            )
        object.__setattr__(self, "retained_session_ids", retained)
        object.__setattr__(self, "removed_session_ids", removed)
        object.__setattr__(
            self, "source_allowed_session_ids", source_allowed
        )
        object.__setattr__(
            self, "target_allowed_session_ids", target_allowed
        )

    def to_json(self) -> dict[str, object]:
        return {
            "previous_fraction_ppm": self.previous_fraction_ppm,
            "removed_session_ids": list(self.removed_session_ids),
            "source_allowed_session_ids": list(
                self.source_allowed_session_ids
            ),
            "target_allowed_session_ids": list(
                self.target_allowed_session_ids
            ),
            "retained_layer_mask": self.retained_layer_mask,
            "drain_policy_sha256": self.drain_policy_sha256,
            "request_id": self.request_id,
            "retained_session_ids": list(self.retained_session_ids),
            "source_generation": self.source_generation,
            "source_geometry_sha256": self.source_geometry_sha256,
            "state": self.state,
            "target_generation": self.target_generation,
            "target_geometry_sha256": self.target_geometry_sha256,
        }


@dataclass(frozen=True)
class RequestPlacementBinding:
    base: RequestBasePlacementBinding
    helper_envelope: RequestHelperEnvelopeBinding | None
    helper_attachment: RequestHelperAttachment | None
    fraction_ppm: int

    def __post_init__(self) -> None:
        if not isinstance(self.base, RequestBasePlacementBinding):
            raise ModelPlacementControllerError(
                "request base placement binding is invalid"
            )
        if self.helper_envelope is not None and not isinstance(
            self.helper_envelope, RequestHelperEnvelopeBinding
        ):
            raise ModelPlacementControllerError(
                "request helper envelope binding is invalid"
            )
        if self.helper_attachment is not None and not isinstance(
            self.helper_attachment, RequestHelperAttachment
        ):
            raise ModelPlacementControllerError(
                "request helper attachment is invalid"
            )
        fraction = _integer("request placement fraction", self.fraction_ppm)
        if fraction > 1_000_000:
            raise ModelPlacementControllerError(
                "request placement fraction is invalid"
            )
        if (
            fraction > 0
            and self.helper_envelope is not None
            and self.helper_attachment is None
        ):
            raise ModelPlacementControllerError(
                "assisted request lacks a helper attachment"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "base": self.base.to_json(),
            "fraction_ppm": self.fraction_ppm,
            "helper_attachment": (
                None
                if self.helper_attachment is None
                else self.helper_attachment.to_json()
            ),
            "helper_envelope": (
                None
                if self.helper_envelope is None
                else self.helper_envelope.to_json()
            ),
            "schema": "research-scheduler-request-placement-binding-v1",
        }
