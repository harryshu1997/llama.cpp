"""Execution-plan contracts grouped by responsibility: transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ..runtime_capabilities import ROUTE_MATURITY_STATES
from .common import (
    RUNTIME_TRANSITION_RECEIPT_STATUSES,
    RuntimePlanError,
    _estimation_metadata,
    _integer,
    _sha256,
    _text,
)
from .phone import RuntimePhoneShard


@dataclass(frozen=True)
class RuntimeResidencyEviction:
    model_id: str
    artifact_sha256: str
    device_id: str
    resident_bytes: int
    generation: int
    executor_id: str | None = None
    reclaimable_bytes: int | None = None
    replacement_group: str | None = None
    session_id: str | None = None
    resident_geometry_sha256: str | None = None
    operator_plan_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("model_id", "artifact_sha256", "device_id"):
            _text(f"runtime eviction {name}", getattr(self, name))
        if (
            not self.artifact_sha256.startswith("sha256:")
            or len(self.artifact_sha256) != 71
            or any(
                value not in "0123456789abcdef"
                for value in self.artifact_sha256[7:]
            )
        ):
            raise RuntimePlanError(
                "runtime eviction artifact hash must be SHA-256"
            )
        _integer("runtime eviction resident bytes", self.resident_bytes, 1)
        _integer("runtime eviction generation", self.generation)
        if self.executor_id is not None:
            _text("runtime eviction executor id", self.executor_id)
        if self.replacement_group is not None:
            _text(
                "runtime eviction replacement group",
                self.replacement_group,
            )
        session_values = (
            self.session_id,
            self.resident_geometry_sha256,
            self.operator_plan_sha256,
        )
        if any(value is not None for value in session_values):
            if any(value is None for value in session_values):
                raise RuntimePlanError(
                    "runtime eviction session identity is incomplete"
                )
            _text("runtime eviction session id", self.session_id)
            _sha256(
                "runtime eviction resident geometry",
                self.resident_geometry_sha256,
            )
            _sha256(
                "runtime eviction operator plan",
                self.operator_plan_sha256,
            )
            if self.generation < 1:
                raise RuntimePlanError(
                    "runtime eviction session generation is invalid"
                )
        if self.reclaimable_bytes is not None:
            _integer(
                "runtime eviction reclaimable bytes",
                self.reclaimable_bytes,
                self.resident_bytes,
            )
            if (
                self.reclaimable_bytes > self.resident_bytes
                and self.executor_id is None
            ):
                raise RuntimePlanError(
                    "additional eviction memory requires an executor"
                )

    def to_json(self) -> dict[str, int | str]:
        result = {
            "artifact_sha256": self.artifact_sha256,
            "device_id": self.device_id,
            "generation": self.generation,
            "model_id": self.model_id,
            "resident_bytes": self.resident_bytes,
        }
        if self.executor_id is not None:
            result["executor_id"] = self.executor_id
        if self.reclaimable_bytes is not None:
            result["reclaimable_bytes"] = self.reclaimable_bytes
        if self.replacement_group is not None:
            result["replacement_group"] = self.replacement_group
        if self.session_id is not None:
            result["session_id"] = self.session_id
            result["resident_geometry_sha256"] = (
                self.resident_geometry_sha256
            )
            result["operator_plan_sha256"] = (
                self.operator_plan_sha256
            )
        return result


@dataclass(frozen=True)
class RuntimeTransitionPlan:
    transition_id: str
    device_id: str
    source_state: str
    target_state: str
    latency_us: int
    energy_uj: int
    resource_ids: tuple[str, ...]
    maturity: str
    resource_slots: Mapping[str, int] = field(default_factory=dict)
    evictions: tuple[RuntimeResidencyEviction, ...] = ()
    executor_id: str | None = None
    prepares_device_ids: tuple[str, ...] = ()
    energy_maturity: str | None = None
    phone_shards: tuple[RuntimePhoneShard, ...] = ()
    changed_phone_session_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "transition_id",
            "device_id",
            "source_state",
            "target_state",
        ):
            _text(f"transition plan {name}", getattr(self, name))
        _integer("transition plan latency", self.latency_us)
        _integer("transition plan energy", self.energy_uj)
        resources = tuple(self.resource_ids)
        if not resources or len(resources) != len(set(resources)):
            raise RuntimePlanError("transition plan resources are invalid")
        for resource_id in resources:
            _text("transition plan resource", resource_id)
        slots = {
            _text("transition plan slot resource", resource_id): _integer(
                "transition plan resource slots", value, 1
            )
            for resource_id, value in self.resource_slots.items()
        }
        if set(slots) - set(resources):
            raise RuntimePlanError(
                "transition plan slots reference an unknown resource"
            )
        slots = {
            resource_id: slots.get(resource_id, 1)
            for resource_id in resources
        }
        if self.maturity not in ROUTE_MATURITY_STATES:
            raise RuntimePlanError("transition plan maturity is invalid")
        energy_maturity = (
            self.maturity
            if self.energy_maturity is None else self.energy_maturity
        )
        if energy_maturity not in ROUTE_MATURITY_STATES:
            raise RuntimePlanError(
                "transition plan energy maturity is invalid"
            )
        if self.executor_id is not None:
            _text("transition plan executor id", self.executor_id)
        prepares = self.prepares_device_ids or (self.device_id,)
        if (
            len(prepares) != len(set(prepares))
            or any(
                type(value) is not str
                or not value
                or not value.isascii()
                for value in prepares
            )
            or self.device_id not in prepares
        ):
            raise RuntimePlanError(
                "transition plan prepared devices are invalid"
            )
        evictions = tuple(self.evictions)
        if any(
            not isinstance(row, RuntimeResidencyEviction)
            or row.device_id not in prepares
            for row in evictions
        ) or len({
            (
                row.model_id,
                row.artifact_sha256,
                row.device_id,
                row.session_id,
            )
            for row in evictions
        }) != len(evictions):
            raise RuntimePlanError("transition plan evictions are invalid")
        phone_shards = tuple(self.phone_shards)
        covered_layers_by_artifact: dict[str, int] = {}
        if any(
            not isinstance(row, RuntimePhoneShard) for row in phone_shards
        ) or len({
            row.session_id for row in phone_shards
        }) != len(phone_shards):
            raise RuntimePlanError("transition plan phone shards are invalid")
        for shard in phone_shards:
            artifact = shard.artifact_sha256 or "legacy-single-artifact"
            covered_layers = covered_layers_by_artifact.get(artifact, 0)
            if covered_layers & shard.layer_mask:
                raise RuntimePlanError(
                    "transition plan phone shard layers overlap"
                )
            covered_layers_by_artifact[artifact] = (
                covered_layers | shard.layer_mask
            )
        changed_phone_sessions = tuple(sorted(
            _text("changed phone session", row)
            for row in self.changed_phone_session_ids
        ))
        if len(changed_phone_sessions) != len(set(changed_phone_sessions)):
            raise RuntimePlanError(
                "changed phone sessions are duplicated"
            )
        object.__setattr__(self, "resource_ids", tuple(sorted(resources)))
        object.__setattr__(
            self,
            "resource_slots",
            MappingProxyType(dict(sorted(slots.items()))),
        )
        object.__setattr__(
            self,
            "evictions",
            tuple(sorted(evictions, key=lambda row: (
                row.device_id,
                row.model_id,
                row.artifact_sha256,
                "" if row.session_id is None else row.session_id,
            ))),
        )
        object.__setattr__(
            self, "prepares_device_ids", tuple(sorted(prepares))
        )
        object.__setattr__(self, "energy_maturity", energy_maturity)
        object.__setattr__(
            self,
            "phone_shards",
            tuple(sorted(phone_shards, key=lambda row: row.session_id)),
        )
        object.__setattr__(
            self, "changed_phone_session_ids", changed_phone_sessions
        )

    def to_json(self) -> dict[str, object]:
        result = {
            "device_id": self.device_id,
            "energy_uj": self.energy_uj,
            "latency_us": self.latency_us,
            "maturity": self.maturity,
            "resource_ids": list(self.resource_ids),
            "source_state": self.source_state,
            "target_state": self.target_state,
            "transition_id": self.transition_id,
        }
        if any(value != 1 for value in self.resource_slots.values()):
            result["resource_slots"] = dict(self.resource_slots)
        if self.evictions:
            result["evictions"] = [row.to_json() for row in self.evictions]
        if self.executor_id is not None:
            result["executor_id"] = self.executor_id
        if self.prepares_device_ids != (self.device_id,):
            result["prepares_device_ids"] = list(
                self.prepares_device_ids
            )
        if self.energy_maturity != self.maturity:
            result["energy_maturity"] = self.energy_maturity
        if self.phone_shards:
            result["phone_shards"] = [
                row.to_json() for row in self.phone_shards
            ]
        if self.changed_phone_session_ids:
            result["changed_phone_session_ids"] = list(
                self.changed_phone_session_ids
            )
        return result


@dataclass(frozen=True)
class RuntimeTransitionReceipt:
    ticket_id: str
    request_id: str
    artifact_sha256: str
    operator_plan_sha256: str
    transition_id: str
    executor_id: str
    endpoint: str
    device_id: str
    source_state: str
    target_state: str
    resource_ids: tuple[str, ...]
    started_us: int
    finished_us: int
    status: str
    resource_slots: Mapping[str, int] = field(default_factory=dict)
    evicted_artifact_sha256s: tuple[str, ...] = ()
    energy_boundary_id: str | None = None
    fleet_energy_uj_by_domain: Mapping[str, int] = field(
        default_factory=dict
    )
    transfer_energy_uj_by_link: Mapping[str, int] = field(
        default_factory=dict
    )
    measurement_evidence_ids: tuple[str, ...] = ()
    energy_attribution_kind: str | None = None
    energy_estimation_metadata: Mapping[
        str, int | str | bool
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "ticket_id",
            "request_id",
            "artifact_sha256",
            "operator_plan_sha256",
            "transition_id",
            "executor_id",
            "endpoint",
            "device_id",
            "source_state",
            "target_state",
        ):
            _text(f"transition receipt {name}", getattr(self, name))
        for name in ("artifact_sha256", "operator_plan_sha256"):
            value = getattr(self, name)
            if (
                not value.startswith("sha256:")
                or len(value) != 71
                or any(character not in "0123456789abcdef" for character in value[7:])
            ):
                raise RuntimePlanError(
                    f"transition receipt {name} must be SHA-256"
                )
        resources = tuple(self.resource_ids)
        if not resources or len(resources) != len(set(resources)):
            raise RuntimePlanError(
                "transition receipt resources are invalid"
            )
        for resource_id in resources:
            _text("transition receipt resource", resource_id)
        slots = {
            _text("transition receipt slot resource", resource_id): _integer(
                "transition receipt resource slots", value, 1
            )
            for resource_id, value in self.resource_slots.items()
        }
        if set(slots) - set(resources):
            raise RuntimePlanError(
                "transition receipt slots reference an unknown resource"
            )
        slots = {
            resource_id: slots.get(resource_id, 1)
            for resource_id in resources
        }
        _integer("transition receipt started", self.started_us)
        _integer("transition receipt finished", self.finished_us)
        if self.finished_us < self.started_us:
            raise RuntimePlanError(
                "transition receipt finishes before it starts"
            )
        if self.status not in RUNTIME_TRANSITION_RECEIPT_STATUSES:
            raise RuntimePlanError("transition receipt status is invalid")
        evicted = tuple(sorted(self.evicted_artifact_sha256s))
        if len(evicted) != len(set(evicted)) or any(
            not value.startswith("sha256:")
            or len(value) != 71
            or any(
                character not in "0123456789abcdef"
                for character in value[7:]
            )
            for value in evicted
        ):
            raise RuntimePlanError(
                "transition receipt evictions are invalid"
            )
        object.__setattr__(self, "resource_ids", tuple(sorted(resources)))
        object.__setattr__(
            self,
            "resource_slots",
            MappingProxyType(dict(sorted(slots.items()))),
        )
        object.__setattr__(
            self, "evicted_artifact_sha256s", evicted
        )
        fleet_energy = {
            _text("transition receipt energy domain", key): _integer(
                "transition receipt domain energy", value
            )
            for key, value in self.fleet_energy_uj_by_domain.items()
        }
        transfer_energy = {
            _text("transition receipt transfer link", key): _integer(
                "transition receipt transfer energy", value
            )
            for key, value in self.transfer_energy_uj_by_link.items()
        }
        evidence = tuple(
            _text("transition receipt measurement evidence", value)
            for value in self.measurement_evidence_ids
        )
        if self.energy_boundary_id is None:
            if (
                fleet_energy or transfer_energy or evidence
                or self.energy_attribution_kind is not None
                or self.energy_estimation_metadata
            ):
                raise RuntimePlanError(
                    "transition receipt measurement lacks an energy boundary"
                )
        else:
            _text(
                "transition receipt energy boundary",
                self.energy_boundary_id,
            )
            if not fleet_energy or not evidence:
                raise RuntimePlanError(
                    "transition receipt energy measurement is incomplete"
                )
            if self.energy_attribution_kind is None:
                object.__setattr__(
                    self, "energy_attribution_kind", "diagnostic"
                )
            if self.energy_attribution_kind not in {
                "diagnostic", "isolated", "matched_abba", "device_domain"
            }:
                raise RuntimePlanError(
                    "transition receipt energy attribution is invalid"
                )
        object.__setattr__(
            self,
            "fleet_energy_uj_by_domain",
            MappingProxyType(dict(sorted(fleet_energy.items()))),
        )
        object.__setattr__(
            self,
            "transfer_energy_uj_by_link",
            MappingProxyType(dict(sorted(transfer_energy.items()))),
        )
        object.__setattr__(
            self,
            "measurement_evidence_ids",
            tuple(sorted(evidence)),
        )
        object.__setattr__(
            self,
            "energy_estimation_metadata",
            _estimation_metadata(self.energy_estimation_metadata),
        )

    @property
    def actual_latency_us(self) -> int:
        return self.finished_us - self.started_us

    @property
    def whole_fleet_energy_uj(self) -> int | None:
        if self.energy_boundary_id is None:
            return None
        return sum(self.fleet_energy_uj_by_domain.values())

    def to_json(self) -> dict[str, object]:
        result = {
            "artifact_sha256": self.artifact_sha256,
            "device_id": self.device_id,
            "endpoint": self.endpoint,
            "executor_id": self.executor_id,
            "finished_us": self.finished_us,
            "operator_plan_sha256": self.operator_plan_sha256,
            "request_id": self.request_id,
            "resource_ids": list(self.resource_ids),
            "source_state": self.source_state,
            "started_us": self.started_us,
            "status": self.status,
            "target_state": self.target_state,
            "ticket_id": self.ticket_id,
            "transition_id": self.transition_id,
        }
        if any(value != 1 for value in self.resource_slots.values()):
            result["resource_slots"] = dict(self.resource_slots)
        if self.evicted_artifact_sha256s:
            result["evicted_artifact_sha256s"] = list(
                self.evicted_artifact_sha256s
            )
        if self.energy_boundary_id is not None:
            result.update({
                "energy_attribution_kind": self.energy_attribution_kind,
                "energy_boundary_id": self.energy_boundary_id,
                "fleet_energy_uj_by_domain": dict(
                    self.fleet_energy_uj_by_domain
                ),
                "measurement_evidence_ids": list(
                    self.measurement_evidence_ids
                ),
                "energy_estimation_metadata": dict(
                    self.energy_estimation_metadata
                ),
                "transfer_energy_uj_by_link": dict(
                    self.transfer_energy_uj_by_link
                ),
                "whole_fleet_energy_uj": self.whole_fleet_energy_uj,
            })
        return result
