"""Model demand, layout and request-binding records: layout."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..phone_shards import PhoneFfnResidencyLayout, PhoneFfnSessionIdentity
from ..types import canonical_sha256
from .common import (
    ModelPlacementControllerError,
    PHONE_RESIDENCY_LAYOUT_STATES,
    PHONE_SESSION_RESIDENCY_STATES,
    _identities,
    _integer,
    _optional_sha256,
    _sha256,
    _text,
)


@dataclass(frozen=True)
class PhoneLayoutRequestImpact:
    """Planning estimates only; never an execution or qualification authority."""

    request_id: str
    remaining_tokens: int
    current_layer_mask: int
    retained_layer_mask: int
    evidence_reused: bool
    verification_feasible: bool
    verification_tokens: int = 0
    verification_us: int = 0
    retained_assistance_loss_uj: int = 0
    verification_overhead_uj: int = 0
    reason: str = "EXACT_EXECUTION_RETAINED"

    def __post_init__(self) -> None:
        _text("phone layout affected request", self.request_id)
        _text("phone layout impact reason", self.reason)
        for name in ("remaining_tokens", "current_layer_mask", "retained_layer_mask",
                     "verification_tokens", "verification_us",
                     "retained_assistance_loss_uj", "verification_overhead_uj"):
            _integer("phone layout impact " + name, getattr(self, name))
        if (type(self.evidence_reused) is not bool
                or type(self.verification_feasible) is not bool
                or self.retained_layer_mask & self.current_layer_mask != self.retained_layer_mask):
            raise ModelPlacementControllerError("phone layout request impact is invalid")

    @property
    def incremental_cost_uj(self) -> int:
        # Removed-session lost benefit is already in the session objective.
        return self.retained_assistance_loss_uj + self.verification_overhead_uj

    def to_json(self) -> dict[str, object]:
        return {**asdict(self), "incremental_cost_uj": self.incremental_cost_uj,
                "cost_kind": "estimated_retained_revalidation",
                "removed_session_benefit_counted_in": "session_objective"}


@dataclass(frozen=True)
class PhoneSessionMarginalGain:
    session_id: str
    current_artifact_sha256: str | None
    proposed_artifact_sha256: str | None
    current_geometry_sha256: str | None
    proposed_geometry_sha256: str | None
    remaining_work: int
    current_warm_energy_saved_uj: int
    proposed_warm_energy_saved_uj: int
    session_load_energy_uj: int
    session_eviction_energy_uj: int
    session_replacement_energy_uj: int
    latency_penalty_uj: int
    interference_energy_uj: int
    safety_margin_uj: int
    transition_latency_us: int
    break_even_work: int
    break_even_interval_us: int
    minimum_residency_us: int
    projected_gain_uj: int
    gain_over_current_uj: int
    retained_revalidation_cost_uj: int = 0

    def __post_init__(self) -> None:
        _text("phone session marginal identity", self.session_id)
        _optional_sha256(
            "phone session current artifact",
            self.current_artifact_sha256,
        )
        _optional_sha256(
            "phone session proposed artifact",
            self.proposed_artifact_sha256,
        )
        _optional_sha256(
            "phone session current geometry",
            self.current_geometry_sha256,
        )
        _optional_sha256(
            "phone session proposed geometry",
            self.proposed_geometry_sha256,
        )
        for name in (
            "remaining_work",
            "current_warm_energy_saved_uj",
            "proposed_warm_energy_saved_uj",
            "session_load_energy_uj",
            "session_eviction_energy_uj",
            "session_replacement_energy_uj",
            "latency_penalty_uj",
            "interference_energy_uj",
            "safety_margin_uj",
            "transition_latency_us",
            "break_even_work",
            "break_even_interval_us",
            "minimum_residency_us",
            "retained_revalidation_cost_uj",
        ):
            _integer("phone session " + name, getattr(self, name))
        if type(self.projected_gain_uj) is not int or type(
            self.gain_over_current_uj
        ) is not int:
            raise ModelPlacementControllerError(
                "phone session marginal gain is invalid"
            )

    def to_json(self) -> dict[str, object]:
        return {
            **({"retained_revalidation_cost_uj": self.retained_revalidation_cost_uj}
               if self.retained_revalidation_cost_uj else {}),
            "current_artifact_sha256": self.current_artifact_sha256,
            "current_geometry_sha256": self.current_geometry_sha256,
            "current_warm_energy_saved_uj": (
                self.current_warm_energy_saved_uj
            ),
            "gain_over_current_uj": self.gain_over_current_uj,
            "break_even_interval_us": self.break_even_interval_us,
            "break_even_work": self.break_even_work,
            "interference_energy_uj": self.interference_energy_uj,
            "latency_penalty_uj": self.latency_penalty_uj,
            "minimum_residency_us": self.minimum_residency_us,
            "projected_gain_uj": self.projected_gain_uj,
            "proposed_artifact_sha256": self.proposed_artifact_sha256,
            "proposed_geometry_sha256": self.proposed_geometry_sha256,
            "proposed_warm_energy_saved_uj": (
                self.proposed_warm_energy_saved_uj
            ),
            "remaining_work": self.remaining_work,
            "session_id": self.session_id,
            "session_load_energy_uj": self.session_load_energy_uj,
            "session_eviction_energy_uj": (
                self.session_eviction_energy_uj
            ),
            "session_replacement_energy_uj": (
                self.session_replacement_energy_uj
            ),
            "safety_margin_uj": self.safety_margin_uj,
            "transition_latency_us": self.transition_latency_us,
        }


@dataclass(frozen=True)
class PhoneSessionResidencyState:
    """Controller authority for one physical HTP residency session."""

    session_id: str
    resident_artifact_sha256: str | None
    shard_geometry_sha256: str | None
    operator_plan_sha256: str | None
    session_generation: int
    state: str
    active_helper_references: tuple[str, ...]
    minimum_resident_until_us: int
    replacement_cost_uj: int
    resident_bytes: int
    endpoint: str

    def __post_init__(self) -> None:
        _text("phone session identity", self.session_id)
        if self.state not in PHONE_SESSION_RESIDENCY_STATES:
            raise ModelPlacementControllerError(
                "phone session residency state is invalid"
            )
        references = _identities(
            "phone session helper reference",
            self.active_helper_references,
        )
        _integer(
            "phone session minimum residency",
            self.minimum_resident_until_us,
        )
        _integer(
            "phone session replacement cost", self.replacement_cost_uj
        )
        _text("phone session endpoint", self.endpoint)
        object.__setattr__(self, "active_helper_references", references)
        identity_values = (
            self.resident_artifact_sha256,
            self.shard_geometry_sha256,
            self.operator_plan_sha256,
        )
        identity_absent = all(value is None for value in identity_values)
        if self.state == "EMPTY":
            if (
                not identity_absent
                or self.session_generation != 0
                or self.resident_bytes != 0
                or references
            ):
                raise ModelPlacementControllerError(
                    "empty phone session carries residency"
                )
            return
        if self.state == "UNAVAILABLE" and identity_absent:
            _integer(
                "unavailable phone session generation",
                self.session_generation,
            )
            if self.resident_bytes != 0 or references:
                raise ModelPlacementControllerError(
                    "unavailable empty phone session carries residency"
                )
            return
        if any(value is None for value in identity_values):
            raise ModelPlacementControllerError(
                "phone session resident identity is incomplete"
            )
        _integer("phone session resident bytes", self.resident_bytes, 1)
        identity = PhoneFfnSessionIdentity(
            session_id=self.session_id,
            artifact_sha256=self.resident_artifact_sha256,
            resident_geometry_sha256=self.shard_geometry_sha256,
            operator_plan_sha256=self.operator_plan_sha256,
            session_generation=self.session_generation,
        )
        if identity.session_id != self.session_id:
            raise ModelPlacementControllerError(
                "phone session identity differs"
            )

    @property
    def identity(self) -> PhoneFfnSessionIdentity:
        if self.resident_artifact_sha256 is None:
            raise ModelPlacementControllerError(
                "phone session has no resident identity"
            )
        return PhoneFfnSessionIdentity(
            session_id=self.session_id,
            artifact_sha256=self.resident_artifact_sha256,
            resident_geometry_sha256=self.shard_geometry_sha256,
            operator_plan_sha256=self.operator_plan_sha256,
            session_generation=self.session_generation,
        )

    def matches(self, identity: PhoneFfnSessionIdentity) -> bool:
        return (
            self.resident_artifact_sha256 is not None
            and self.identity == identity
        )

    def to_json(self) -> dict[str, object]:
        return {
            "active_helper_references": list(
                self.active_helper_references
            ),
            "endpoint": self.endpoint,
            "minimum_resident_until_us": (
                self.minimum_resident_until_us
            ),
            "operator_plan_sha256": self.operator_plan_sha256,
            "replacement_cost_uj": self.replacement_cost_uj,
            "resident_artifact_sha256": (
                self.resident_artifact_sha256
            ),
            "resident_bytes": self.resident_bytes,
            "session_generation": self.session_generation,
            "session_id": self.session_id,
            "shard_geometry_sha256": self.shard_geometry_sha256,
            "state": self.state,
        }


@dataclass(frozen=True)
class ModelPhoneResidencyLayout:
    """One controller-owned phone residency lifecycle generation."""

    generation: int
    state: str
    layout: PhoneFfnResidencyLayout
    workspace_bytes: int
    shared_compute_resource_id: str
    shared_transport_resource_ids: tuple[str, ...]
    proposed_at_us: int
    minimum_resident_until_us: int
    minimum_residency_interval_us: int
    session_identities: tuple[PhoneFfnSessionIdentity, ...]
    transition_ticket_id: str | None = None
    transition_ids: tuple[str, ...] = ()
    ready_at_us: int | None = None
    projection_token_sha256: str | None = None
    verification_sha256: str | None = None
    resident_component_identity_sha256: str = ""
    selection_reason: str | None = None
    queue_work_by_artifact: tuple[tuple[str, int], ...] = ()
    queue_benefit_uj: int | None = None
    transition_cost_uj: int | None = None
    switching_margin_uj: int | None = None

    def __post_init__(self) -> None:
        _integer("phone layout generation", self.generation, 1)
        if self.state not in PHONE_RESIDENCY_LAYOUT_STATES:
            raise ModelPlacementControllerError(
                "phone layout state is invalid"
            )
        if not isinstance(self.layout, PhoneFfnResidencyLayout):
            raise ModelPlacementControllerError(
                "phone residency layout is invalid"
            )
        _integer("phone layout workspace", self.workspace_bytes)
        compute = _text(
            "phone layout shared compute resource",
            self.shared_compute_resource_id,
        )
        transports = tuple(sorted(
            _text("phone layout shared transport resource", value)
            for value in self.shared_transport_resource_ids
        ))
        if (
            not transports
            or compute in transports
            or len(transports) != len(set(transports))
        ):
            raise ModelPlacementControllerError(
                "phone layout shared resources are invalid"
            )
        _integer("phone layout proposal time", self.proposed_at_us)
        _integer(
            "phone layout minimum residency",
            self.minimum_resident_until_us,
        )
        _integer(
            "phone layout minimum residency interval",
            self.minimum_residency_interval_us,
        )
        session_identities = tuple(sorted(
            self.session_identities,
            key=lambda row: row.session_id,
        ))
        if (
            any(
                not isinstance(row, PhoneFfnSessionIdentity)
                for row in session_identities
            )
            or len({row.session_id for row in session_identities})
                != len(session_identities)
            or {row.session_id for row in session_identities}
                != {row.session_id for row in self.layout.shards}
            or self.layout.session_generation_by_id != {
                row.session_id: row.session_generation
                for row in session_identities
            }
        ):
            raise ModelPlacementControllerError(
                "phone layout session identities are invalid"
            )
        shard_by_session = {
            row.session_id: row for row in self.layout.shards
        }
        if any(
            identity.artifact_sha256
                != shard_by_session[identity.session_id].artifact_sha256
            or identity.resident_geometry_sha256
                != shard_by_session[identity.session_id]
                    .resident_geometry_sha256
            or identity.operator_plan_sha256
                != shard_by_session[identity.session_id]
                    .operator_plan_sha256
            for identity in session_identities
        ):
            raise ModelPlacementControllerError(
                "phone layout session identity differs from its shard"
            )
        queue_work = tuple(sorted(
            (
                _sha256("phone layout queue artifact", artifact),
                _integer("phone layout queued work", work, 1),
            )
            for artifact, work in self.queue_work_by_artifact
        ))
        if len(queue_work) != len({artifact for artifact, _ in queue_work}):
            raise ModelPlacementControllerError(
                "phone layout queue artifacts are duplicated"
            )
        authorization_costs = (
            self.queue_benefit_uj,
            self.transition_cost_uj,
            self.switching_margin_uj,
        )
        if self.selection_reason is None:
            if queue_work or any(
                value is not None for value in authorization_costs
            ):
                raise ModelPlacementControllerError(
                    "phone layout selection evidence is incomplete"
                )
        else:
            _text("phone layout selection reason", self.selection_reason)
            if not queue_work or any(
                value is None for value in authorization_costs
            ):
                raise ModelPlacementControllerError(
                    "phone layout selection evidence is incomplete"
                )
            for name, value in zip(
                (
                    "queue benefit",
                    "transition cost",
                    "switching margin",
                ),
                authorization_costs,
            ):
                _integer("phone layout " + name, value)
        transition_ids = tuple(sorted(
            _text("phone layout transition", value)
            for value in self.transition_ids
        ))
        if len(transition_ids) != len(set(transition_ids)):
            raise ModelPlacementControllerError(
                "phone layout transitions are duplicated"
            )
        transition_fields = (
            self.transition_ticket_id,
            self.projection_token_sha256,
        )
        if self.state == "PROPOSED":
            if transition_ids or any(
                value is not None for value in transition_fields
            ) or self.ready_at_us is not None \
                    or self.verification_sha256 is not None:
                raise ModelPlacementControllerError(
                    "proposed phone layout carries transition state"
                )
        elif self.state == "PREPARING":
            if (
                not transition_ids
                or any(value is None for value in transition_fields)
                or self.ready_at_us is None
                or self.verification_sha256 is not None
            ):
                raise ModelPlacementControllerError(
                    "active phone layout lacks transition state"
                )
            _text(
                "phone layout transition ticket",
                self.transition_ticket_id,
            )
            _integer("phone layout ready time", self.ready_at_us)
            _sha256(
                "phone layout projection token",
                self.projection_token_sha256,
            )
        else:
            has_transition = bool(transition_ids) and not any(
                value is None for value in transition_fields
            ) and self.ready_at_us is not None
            has_verification = (
                self.verification_sha256 is not None
                and not transition_ids
                and all(value is None for value in transition_fields)
                and self.ready_at_us is not None
            )
            if has_verification:
                _sha256(
                    "phone layout readiness verification",
                    self.verification_sha256,
                )
            elif not has_transition:
                raise ModelPlacementControllerError(
                    "ready phone layout lacks physical verification"
                )
        body = {
            "geometry_sha256": self.layout.geometry_sha256,
            "resident_bytes": self.layout.resident_bytes,
            "schema": "research-scheduler-phone-resident-component-v1",
            "shards": [
                {
                    "artifact_sha256": row.artifact_sha256,
                    "resident_geometry_sha256": (
                        row.resident_geometry_sha256
                    ),
                    "session_generation": (
                        self.layout.session_generation_by_id[
                            row.session_id
                        ]
                    ),
                    "session_id": row.session_id,
                }
                for row in self.layout.shards
            ],
            "shared_compute_resource_id": compute,
            "shared_transport_resource_ids": list(transports),
            "workspace_bytes": self.workspace_bytes,
        }
        expected_component = canonical_sha256(body)
        if (
            self.resident_component_identity_sha256
            and self.resident_component_identity_sha256
                != expected_component
        ):
            raise ModelPlacementControllerError(
                "phone resident component identity differs"
            )
        object.__setattr__(
            self, "shared_transport_resource_ids", transports
        )
        object.__setattr__(self, "session_identities", session_identities)
        object.__setattr__(self, "transition_ids", transition_ids)
        object.__setattr__(self, "queue_work_by_artifact", queue_work)
        object.__setattr__(
            self,
            "resident_component_identity_sha256",
            expected_component,
        )

    @property
    def covered_artifact_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted({
            row.artifact_sha256 for row in self.layout.shards
        }))

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(row.session_id for row in self.layout.shards)

    def covers_artifact(self, artifact_sha256: str) -> bool:
        return artifact_sha256 in self.covered_artifact_sha256s

    def to_json(self) -> dict[str, object]:
        return {
            "covered_artifact_sha256s": list(
                self.covered_artifact_sha256s
            ),
            "generation": self.generation,
            "geometry_sha256": self.layout.geometry_sha256,
            "layout": self.layout.to_json(),
            "minimum_resident_until_us": (
                self.minimum_resident_until_us
            ),
            "minimum_residency_interval_us": (
                self.minimum_residency_interval_us
            ),
            "projection_token_sha256": self.projection_token_sha256,
            "proposed_at_us": self.proposed_at_us,
            "queue_benefit_uj": self.queue_benefit_uj,
            "queue_work_by_artifact": dict(
                self.queue_work_by_artifact
            ),
            "ready_at_us": self.ready_at_us,
            "resident_bytes": self.layout.resident_bytes,
            "resident_component_identity_sha256": (
                self.resident_component_identity_sha256
            ),
            "schema": "research-scheduler-phone-residency-lifecycle-v1",
            "session_identities": [
                row.to_json() for row in self.session_identities
            ],
            "session_ids": list(self.session_ids),
            "shared_compute_resource_id": (
                self.shared_compute_resource_id
            ),
            "shared_transport_resource_ids": list(
                self.shared_transport_resource_ids
            ),
            "selection_reason": self.selection_reason,
            "state": self.state,
            "switching_margin_uj": self.switching_margin_uj,
            "transition_cost_uj": self.transition_cost_uj,
            "transition_ids": list(self.transition_ids),
            "transition_ticket_id": self.transition_ticket_id,
            "verification_sha256": self.verification_sha256,
            "workspace_bytes": self.workspace_bytes,
        }
