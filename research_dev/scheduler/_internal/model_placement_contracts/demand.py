"""Model demand, layout and request-binding records: demand."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..types import canonical_sha256
from .common import (
    MODEL_PLACEMENT_ACTIONS,
    ModelPlacementControllerError,
    _identities,
    _integer,
    _material_bucket,
    _optional_sha256,
    _sha256,
    _text,
)


@dataclass(frozen=True)
class ModelPlacementPolicy:
    debounce_us: int = 250_000
    switch_hysteresis_us: int = 2_000_000
    phone_layout_confirmation_snapshots: int = 3
    phone_session_hysteresis_uj: int = 0
    phone_session_latency_penalty_uj: int = 0
    phone_session_interference_energy_uj: int = 0
    phone_session_safety_margin_uj: int = 0
    phone_minimum_residency_us: int = 30_000_000
    maximum_events: int = 4096

    def __post_init__(self) -> None:
        _integer("model placement debounce", self.debounce_us)
        _integer(
            "model placement switch hysteresis",
            self.switch_hysteresis_us,
        )
        _integer(
            "phone layout confirmation snapshots",
            self.phone_layout_confirmation_snapshots,
            1,
        )
        _integer(
            "phone session hysteresis",
            self.phone_session_hysteresis_uj,
        )
        _integer(
            "phone session latency penalty",
            self.phone_session_latency_penalty_uj,
        )
        _integer(
            "phone session interference energy",
            self.phone_session_interference_energy_uj,
        )
        _integer(
            "phone session safety margin",
            self.phone_session_safety_margin_uj,
        )
        _integer(
            "phone minimum residency", self.phone_minimum_residency_us
        )
        _integer("model placement event capacity", self.maximum_events, 1)


@dataclass(frozen=True)
class ModelDemandSnapshot:
    artifact_sha256: str
    observed_at_us: int
    active_request_count: int
    queued_request_count: int
    queued_input_tokens: int
    queued_output_tokens: int
    oldest_queued_wait_us: int
    predicted_queue_drain_us: int
    current_resident_component_identity_sha256: str | None
    available_device_ids: tuple[str, ...]
    available_session_ids: tuple[str, ...]
    memory_generation_sha256: str
    resource_calendar_generation_sha256: str
    capability_generation_sha256: str
    profile_generation_sha256: str
    transport_generation_sha256: str
    residency_generation_sha256: str
    learning_generation_sha256: str

    def __post_init__(self) -> None:
        _sha256("model demand artifact", self.artifact_sha256)
        _integer("model demand observation time", self.observed_at_us)
        for name in (
            "active_request_count",
            "queued_request_count",
            "queued_input_tokens",
            "queued_output_tokens",
            "oldest_queued_wait_us",
            "predicted_queue_drain_us",
        ):
            _integer("model demand " + name, getattr(self, name))
        _optional_sha256(
            "model demand resident component",
            self.current_resident_component_identity_sha256,
        )
        object.__setattr__(
            self,
            "available_device_ids",
            _identities("model demand device", self.available_device_ids),
        )
        object.__setattr__(
            self,
            "available_session_ids",
            _identities("model demand session", self.available_session_ids),
        )
        for name in (
            "memory_generation_sha256",
            "resource_calendar_generation_sha256",
            "capability_generation_sha256",
            "profile_generation_sha256",
            "transport_generation_sha256",
            "residency_generation_sha256",
            "learning_generation_sha256",
        ):
            _sha256("model demand " + name, getattr(self, name))

    @property
    def pressure_bucket(self) -> str:
        return ":".join((
            "a" + str(_material_bucket(self.active_request_count)),
            "q" + str(_material_bucket(self.queued_request_count)),
            "i" + str(_material_bucket(self.queued_input_tokens)),
            "o" + str(_material_bucket(self.queued_output_tokens)),
            "w" + str(_material_bucket(self.oldest_queued_wait_us)),
            "d" + str(_material_bucket(self.predicted_queue_drain_us)),
        ))

    @property
    def demand_generation_sha256(self) -> str:
        return canonical_sha256({
            "artifact_sha256": self.artifact_sha256,
            "available_device_ids": self.available_device_ids,
            "available_session_ids": self.available_session_ids,
            "capability_generation_sha256": (
                self.capability_generation_sha256
            ),
            "current_resident_component_identity_sha256": (
                self.current_resident_component_identity_sha256
            ),
            "learning_generation_sha256": (
                self.learning_generation_sha256
            ),
            "memory_generation_sha256": self.memory_generation_sha256,
            "pressure_bucket": self.pressure_bucket,
            "profile_generation_sha256": self.profile_generation_sha256,
            "residency_generation_sha256": (
                self.residency_generation_sha256
            ),
            "resource_calendar_generation_sha256": (
                self.resource_calendar_generation_sha256
            ),
            "schema": "research-scheduler-model-demand-generation-v1",
            "transport_generation_sha256": (
                self.transport_generation_sha256
            ),
        })

    def to_json(self) -> dict[str, object]:
        return {
            "active_request_count": self.active_request_count,
            "artifact_sha256": self.artifact_sha256,
            "available_device_ids": list(self.available_device_ids),
            "available_session_ids": list(self.available_session_ids),
            "capability_generation_sha256": (
                self.capability_generation_sha256
            ),
            "current_resident_component_identity_sha256": (
                self.current_resident_component_identity_sha256
            ),
            "demand_generation_sha256": self.demand_generation_sha256,
            "learning_generation_sha256": self.learning_generation_sha256,
            "memory_generation_sha256": self.memory_generation_sha256,
            "observed_at_us": self.observed_at_us,
            "oldest_queued_wait_us": self.oldest_queued_wait_us,
            "predicted_queue_drain_us": self.predicted_queue_drain_us,
            "pressure_bucket": self.pressure_bucket,
            "profile_generation_sha256": self.profile_generation_sha256,
            "queued_input_tokens": self.queued_input_tokens,
            "queued_output_tokens": self.queued_output_tokens,
            "queued_request_count": self.queued_request_count,
            "residency_generation_sha256": (
                self.residency_generation_sha256
            ),
            "resource_calendar_generation_sha256": (
                self.resource_calendar_generation_sha256
            ),
            "schema": "research-scheduler-model-demand-snapshot-v1",
            "transport_generation_sha256": (
                self.transport_generation_sha256
            ),
        }


@dataclass(frozen=True)
class ModelPlacementTrigger:
    snapshot: ModelDemandSnapshot
    epoch_sha256: str | None = None
    epoch_demand_generation_sha256: str | None = None
    epoch_pressure_bucket: str | None = None
    epoch_valid_until_us: int | None = None
    selected_route_feasible: bool = True
    notification_reasons: tuple[str, ...] = ()
    old_component_identity_sha256: str | None = None
    proposed_component_identity_sha256: str | None = None
    old_warm_energy_lower_uj: int | None = None
    old_warm_energy_upper_uj: int | None = None
    new_warm_energy_upper_uj: int | None = None
    new_transition_energy_upper_uj: int | None = None
    old_restore_energy_upper_uj: int | None = None
    desktop_latency_upper_us: int | None = None
    proposed_latency_upper_us: int | None = None
    maximum_latency_ppm: int = 1_000_000
    predicted_reuse_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, ModelDemandSnapshot):
            raise ModelPlacementControllerError(
                "model placement trigger snapshot is invalid"
            )
        _optional_sha256("model placement epoch", self.epoch_sha256)
        _optional_sha256(
            "model placement epoch demand generation",
            self.epoch_demand_generation_sha256,
        )
        if self.epoch_pressure_bucket is not None:
            _text(
                "model placement epoch pressure bucket",
                self.epoch_pressure_bucket,
            )
        if self.epoch_valid_until_us is not None:
            _integer(
                "model placement epoch validity",
                self.epoch_valid_until_us,
            )
        if type(self.selected_route_feasible) is not bool:
            raise ModelPlacementControllerError(
                "model placement feasibility flag is invalid"
            )
        reasons = tuple(sorted(
            _text("model placement notification", value)
            for value in self.notification_reasons
        ))
        if len(reasons) != len(set(reasons)):
            raise ModelPlacementControllerError(
                "model placement notifications are duplicated"
            )
        object.__setattr__(self, "notification_reasons", reasons)
        for name in (
            "old_component_identity_sha256",
            "proposed_component_identity_sha256",
        ):
            _optional_sha256("model placement " + name, getattr(self, name))
        cost_values = (
            self.new_warm_energy_upper_uj,
            self.new_transition_energy_upper_uj,
            self.old_restore_energy_upper_uj,
        )
        old_energy_values = (
            self.old_warm_energy_lower_uj,
            self.old_warm_energy_upper_uj,
        )
        if any(value is not None for value in cost_values + old_energy_values):
            if (
                any(value is None for value in cost_values)
                or all(value is None for value in old_energy_values)
            ):
                raise ModelPlacementControllerError(
                    "model placement transition energy is incomplete"
                )
            for value in cost_values + old_energy_values:
                if value is None:
                    continue
                _integer("model placement transition energy", value)
            if (
                self.old_warm_energy_lower_uj is not None
                and self.old_warm_energy_upper_uj is not None
                and self.old_warm_energy_lower_uj
                    > self.old_warm_energy_upper_uj
            ):
                raise ModelPlacementControllerError(
                    "model placement old warm energy bounds are invalid"
                )
        latency_values = (
            self.desktop_latency_upper_us,
            self.proposed_latency_upper_us,
        )
        if any(value is not None for value in latency_values):
            if any(value is None for value in latency_values):
                raise ModelPlacementControllerError(
                    "model placement latency comparison is incomplete"
                )
            for value in latency_values:
                _integer("model placement latency", value, 1)
        _integer(
            "model placement maximum latency ppm",
            self.maximum_latency_ppm,
            1,
        )
        _integer(
            "model placement predicted reuse",
            self.predicted_reuse_count,
        )

    @property
    def expected_reuse_count(self) -> int:
        return max(
            1,
            self.snapshot.active_request_count
            + self.snapshot.queued_request_count
            + self.predicted_reuse_count,
        )

    @property
    def break_even_use_count(self) -> int | None:
        if self.new_transition_energy_upper_uj is None:
            return None
        assert self.old_restore_energy_upper_uj is not None
        assert self.new_warm_energy_upper_uj is not None
        old_warm_energy = (
            self.old_warm_energy_lower_uj
            if self.old_warm_energy_lower_uj is not None
            else self.old_warm_energy_upper_uj
        )
        assert old_warm_energy is not None
        saving = old_warm_energy - self.new_warm_energy_upper_uj
        if saving <= 0:
            return None
        numerator = (
            self.new_transition_energy_upper_uj
            + self.old_restore_energy_upper_uj
        )
        return math.ceil(numerator / saving)

    @property
    def warm_energy_positive(self) -> bool | None:
        if self.new_warm_energy_upper_uj is None:
            return None
        old_warm_energy = (
            self.old_warm_energy_lower_uj
            if self.old_warm_energy_lower_uj is not None
            else self.old_warm_energy_upper_uj
        )
        assert old_warm_energy is not None
        return self.new_warm_energy_upper_uj < old_warm_energy


@dataclass(frozen=True)
class ModelPlacementAction:
    kind: str
    trigger_reasons: tuple[str, ...]
    demand_generation_sha256: str
    pressure_bucket: str
    expected_reuse_count: int
    break_even_use_count: int | None
    old_component_identity_sha256: str | None
    new_component_identity_sha256: str | None
    selected_transition_energy_uj: int | None
    coalesced: bool = False

    def __post_init__(self) -> None:
        if self.kind not in MODEL_PLACEMENT_ACTIONS:
            raise ModelPlacementControllerError(
                "model placement action is invalid"
            )
        reasons = tuple(sorted(
            _text("model placement trigger reason", value)
            for value in self.trigger_reasons
        ))
        if not reasons or len(reasons) != len(set(reasons)):
            raise ModelPlacementControllerError(
                "model placement trigger reasons are invalid"
            )
        object.__setattr__(self, "trigger_reasons", reasons)
        _sha256(
            "model placement demand generation",
            self.demand_generation_sha256,
        )
        _text("model placement pressure bucket", self.pressure_bucket)
        _integer(
            "model placement expected reuse", self.expected_reuse_count, 1
        )
        if self.break_even_use_count is not None:
            _integer(
                "model placement break-even use count",
                self.break_even_use_count,
            )
        for name in (
            "old_component_identity_sha256",
            "new_component_identity_sha256",
        ):
            _optional_sha256("model placement action " + name, getattr(self, name))
        if self.selected_transition_energy_uj is not None:
            _integer(
                "model placement selected transition energy",
                self.selected_transition_energy_uj,
            )
        if type(self.coalesced) is not bool:
            raise ModelPlacementControllerError(
                "model placement coalesced flag is invalid"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "break_even_use_count": self.break_even_use_count,
            "coalesced": self.coalesced,
            "demand_generation_sha256": self.demand_generation_sha256,
            "expected_reuse_count": self.expected_reuse_count,
            "kind": self.kind,
            "new_component_identity_sha256": (
                self.new_component_identity_sha256
            ),
            "old_component_identity_sha256": (
                self.old_component_identity_sha256
            ),
            "pressure_bucket": self.pressure_bucket,
            "selected_transition_energy_uj": (
                self.selected_transition_energy_uj
            ),
            "trigger_reasons": list(self.trigger_reasons),
        }
