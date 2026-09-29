"""Device, executor, catalog and snapshot capability contracts: transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ..runtime_system_cost import ROUTE_MATURITY_STATES
from .common import (
    RESIDENCY_STATES,
    RuntimeCapabilityError,
    _integer,
    _list,
    _object,
    _text,
    _texts,
)


@dataclass(frozen=True)
class RuntimeTransitionCapability:
    transition_id: str
    device_id: str
    source_state: str
    target_state: str
    fixed_latency_us: int
    bandwidth_bytes_per_s: int
    fixed_energy_uj: int
    dynamic_pj_per_byte: int
    resource_ids: tuple[str, ...]
    maturity: str
    evidence_ids: tuple[str, ...]
    resource_slots: Mapping[str, int] = field(default_factory=dict)
    artifact_sha256: str | None = None
    executor_id: str | None = None
    prepares_device_ids: tuple[str, ...] = ()
    energy_maturity: str | None = None

    def __post_init__(self) -> None:
        _text("transition id", self.transition_id)
        _text("transition device id", self.device_id)
        if self.source_state not in RESIDENCY_STATES or self.target_state not in (
            RESIDENCY_STATES
        ):
            raise RuntimeCapabilityError("transition residency state is invalid")
        if (
            self.source_state == self.target_state
            and self.executor_id is None
        ):
            raise RuntimeCapabilityError(
                "same-state transition requires an executor"
            )
        _integer("transition fixed latency", self.fixed_latency_us)
        _integer("transition bandwidth", self.bandwidth_bytes_per_s, 1)
        _integer("transition fixed energy", self.fixed_energy_uj)
        _integer("transition dynamic energy", self.dynamic_pj_per_byte)
        resources = _texts("transition resource", self.resource_ids)
        slots = {
            _text("transition slot resource", resource_id): _integer(
                "transition resource slots", value, 1
            )
            for resource_id, value in self.resource_slots.items()
        }
        if set(slots) - set(resources):
            raise RuntimeCapabilityError(
                "transition slots reference an unknown resource"
            )
        slots = {
            resource_id: slots.get(resource_id, 1)
            for resource_id in resources
        }
        if self.maturity not in ROUTE_MATURITY_STATES:
            raise RuntimeCapabilityError("transition maturity is invalid")
        energy_maturity = (
            self.maturity
            if self.energy_maturity is None else self.energy_maturity
        )
        if energy_maturity not in ROUTE_MATURITY_STATES:
            raise RuntimeCapabilityError(
                "transition energy maturity is invalid"
            )
        evidence = _texts("transition evidence id", self.evidence_ids)
        if self.artifact_sha256 is not None:
            digest = _text(
                "transition artifact hash", self.artifact_sha256
            )
            if (
                not digest.startswith("sha256:")
                or len(digest) != 71
                or any(value not in "0123456789abcdef" for value in digest[7:])
            ):
                raise RuntimeCapabilityError(
                    "transition artifact hash must be SHA-256"
                )
        if self.executor_id is not None:
            _text("transition executor id", self.executor_id)
        prepares = self.prepares_device_ids or (self.device_id,)
        prepares = _texts("transition prepared device", tuple(prepares))
        if self.device_id not in prepares:
            raise RuntimeCapabilityError(
                "transition anchor device is not prepared"
            )
        object.__setattr__(self, "resource_ids", resources)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "energy_maturity", energy_maturity)
        object.__setattr__(
            self,
            "resource_slots",
            MappingProxyType(dict(sorted(slots.items()))),
        )
        object.__setattr__(
            self, "prepares_device_ids", tuple(sorted(prepares))
        )

    def cost(self, amount_bytes: int) -> tuple[int, int]:
        amount_bytes = _integer("transition bytes", amount_bytes)
        latency = self.fixed_latency_us + (
            amount_bytes * 1_000_000 + self.bandwidth_bytes_per_s - 1
        ) // self.bandwidth_bytes_per_s
        energy = self.fixed_energy_uj + (
            amount_bytes * self.dynamic_pj_per_byte + 999_999
        ) // 1_000_000
        return latency, energy

    def to_json(self) -> dict[str, object]:
        result = {
            "bandwidth_bytes_per_s": self.bandwidth_bytes_per_s,
            "device_id": self.device_id,
            "dynamic_pj_per_byte": self.dynamic_pj_per_byte,
            "evidence_ids": list(self.evidence_ids),
            "fixed_energy_uj": self.fixed_energy_uj,
            "fixed_latency_us": self.fixed_latency_us,
            "maturity": self.maturity,
            "resource_ids": list(self.resource_ids),
            "source_state": self.source_state,
            "target_state": self.target_state,
            "transition_id": self.transition_id,
        }
        if any(value != 1 for value in self.resource_slots.values()):
            result["resource_slots"] = dict(self.resource_slots)
        if self.artifact_sha256 is not None:
            result["artifact_sha256"] = self.artifact_sha256
        if self.executor_id is not None:
            result["executor_id"] = self.executor_id
        if self.prepares_device_ids != (self.device_id,):
            result["prepares_device_ids"] = list(
                self.prepares_device_ids
            )
        if self.energy_maturity != self.maturity:
            result["energy_maturity"] = self.energy_maturity
        return result

    @classmethod
    def from_json(cls, value: object) -> "RuntimeTransitionCapability":
        row = _object("runtime transition capability", value)
        return cls(
            transition_id=row.get("transition_id"),
            device_id=row.get("device_id"),
            source_state=row.get("source_state"),
            target_state=row.get("target_state"),
            fixed_latency_us=row.get("fixed_latency_us"),
            bandwidth_bytes_per_s=row.get("bandwidth_bytes_per_s"),
            fixed_energy_uj=row.get("fixed_energy_uj"),
            dynamic_pj_per_byte=row.get("dynamic_pj_per_byte"),
            resource_ids=tuple(_list(
                "runtime transition resources", row.get("resource_ids")
            )),
            maturity=row.get("maturity"),
            evidence_ids=tuple(_list(
                "runtime transition evidence", row.get("evidence_ids")
            )),
            resource_slots=dict(_object(
                "runtime transition resource slots",
                row.get("resource_slots", {}),
            )),
            artifact_sha256=row.get("artifact_sha256"),
            executor_id=row.get("executor_id"),
            prepares_device_ids=tuple(_list(
                "runtime transition prepared devices",
                row.get("prepares_device_ids", []),
            )),
            energy_maturity=row.get("energy_maturity"),
        )
