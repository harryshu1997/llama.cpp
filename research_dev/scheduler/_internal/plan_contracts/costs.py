"""Execution-plan contracts grouped by responsibility: costs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .common import RuntimePlanError, _integer, _text


@dataclass(frozen=True)
class RuntimeTransferCost:
    step_id: str
    source_device: str
    target_device: str
    payload_bytes: int
    invocations: int
    total_bytes: int
    queue_depth: int
    concurrent_streams: int
    message_waves: int
    fixed_latency_us: int
    latency_us: int
    dynamic_energy_uj: int
    link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("step_id", "source_device", "target_device"):
            _text("runtime transfer " + name, getattr(self, name))
        for name in (
            "payload_bytes",
            "invocations",
            "total_bytes",
            "queue_depth",
            "concurrent_streams",
            "message_waves",
        ):
            _integer("runtime transfer " + name, getattr(self, name), 1)
        for name in ("fixed_latency_us", "latency_us", "dynamic_energy_uj"):
            _integer("runtime transfer " + name, getattr(self, name))
        if self.total_bytes != self.payload_bytes * self.invocations:
            raise RuntimePlanError("runtime transfer byte count differs")
        links = tuple(
            _text("runtime transfer link", value) for value in self.link_ids
        )
        if not links or len(links) != len(set(links)):
            raise RuntimePlanError("runtime transfer links are invalid")
        object.__setattr__(self, "link_ids", links)

    def to_json(self) -> dict[str, object]:
        return {
            "concurrent_streams": self.concurrent_streams,
            "dynamic_energy_uj": self.dynamic_energy_uj,
            "fixed_latency_us": self.fixed_latency_us,
            "invocations": self.invocations,
            "latency_us": self.latency_us,
            "link_ids": list(self.link_ids),
            "message_waves": self.message_waves,
            "payload_bytes": self.payload_bytes,
            "queue_depth": self.queue_depth,
            "source_device": self.source_device,
            "step_id": self.step_id,
            "target_device": self.target_device,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class AutomatedRouteCost:
    start_us: int
    finish_us: int
    finish_upper_us: int
    service_us: int
    service_upper_us: int
    queue_delay_us: int
    compute_us: int
    memory_us: int
    transfer_us: int
    join_wait_us: int
    exposed_tail_us: int
    load_us: int
    eviction_us: int
    restore_us: int
    switching_us: int
    interference_us: int
    fleet_energy_uj: int | None
    fleet_energy_lower_uj: int | None
    fleet_energy_upper_uj: int | None
    component_service_us: int
    component_energy_uj: int | None
    memory_by_resource_bytes: Mapping[str, int]
    warm_execution_energy_uj: int | None = None
    warm_execution_energy_lower_uj: int | None = None
    warm_execution_energy_upper_uj: int | None = None
    transition_energy_uj: int | None = None
    transition_energy_lower_uj: int | None = None
    transition_energy_upper_uj: int | None = None
    residency_hysteresis_us: int = 0
    transfer_costs: tuple[RuntimeTransferCost, ...] = ()
    latency_evidence: str = "ABSENT"
    energy_evidence: str = "ABSENT"

    def __post_init__(self) -> None:
        for name in (
            "start_us",
            "finish_us",
            "finish_upper_us",
            "service_us",
            "service_upper_us",
            "queue_delay_us",
            "compute_us",
            "memory_us",
            "transfer_us",
            "join_wait_us",
            "exposed_tail_us",
            "load_us",
            "eviction_us",
            "restore_us",
            "switching_us",
            "interference_us",
            "component_service_us",
            "residency_hysteresis_us",
        ):
            _integer(f"automated cost {name}", getattr(self, name))
        if (
            self.service_us <= 0
            or self.service_upper_us < self.service_us
            or self.component_service_us <= 0
        ):
            raise RuntimePlanError("automated service bound is invalid")
        if self.finish_us != self.start_us + self.service_us:
            raise RuntimePlanError("automated finish does not match service")
        if self.finish_upper_us != self.start_us + self.service_upper_us:
            raise RuntimePlanError("automated upper finish does not match service")
        for name in (
            "fleet_energy_uj",
            "fleet_energy_lower_uj",
            "fleet_energy_upper_uj",
            "component_energy_uj",
        ):
            value = getattr(self, name)
            if value is not None:
                _integer(f"automated cost {name}", value, 1)
        energy = (
            self.fleet_energy_lower_uj,
            self.fleet_energy_uj,
            self.fleet_energy_upper_uj,
        )
        if any(value is None for value in energy) and not all(
            value is None for value in energy
        ):
            raise RuntimePlanError(
                "automated energy bounds must be all present or all absent"
            )
        if all(value is not None for value in energy) and not (
            energy[0] <= energy[1] <= energy[2]
        ):
            raise RuntimePlanError("automated energy bounds are unordered")
        warm_energy = (
            self.warm_execution_energy_lower_uj,
            self.warm_execution_energy_uj,
            self.warm_execution_energy_upper_uj,
        )
        transition_energy = (
            self.transition_energy_lower_uj,
            self.transition_energy_uj,
            self.transition_energy_upper_uj,
        )
        decomposition = warm_energy + transition_energy
        if any(value is None for value in decomposition) and not all(
            value is None for value in decomposition
        ):
            raise RuntimePlanError(
                "automated energy decomposition must be all present or absent"
            )
        if all(value is not None for value in decomposition):
            for name, value in (
                ("warm execution lower energy", warm_energy[0]),
                ("warm execution energy", warm_energy[1]),
                ("warm execution upper energy", warm_energy[2]),
            ):
                _integer("automated cost " + name, value, 1)
            for name, value in (
                ("transition lower energy", transition_energy[0]),
                ("transition energy", transition_energy[1]),
                ("transition upper energy", transition_energy[2]),
            ):
                _integer("automated cost " + name, value)
            if not (
                warm_energy[0] <= warm_energy[1] <= warm_energy[2]
                and transition_energy[0]
                    <= transition_energy[1]
                    <= transition_energy[2]
            ):
                raise RuntimePlanError(
                    "automated energy decomposition bounds are unordered"
                )
            if any(value is None for value in energy):
                raise RuntimePlanError(
                    "automated energy decomposition lacks fleet bounds"
                )
            if (
                energy[0] != warm_energy[0] + transition_energy[0]
                or energy[1] != warm_energy[1] + transition_energy[1]
                or energy[2] != warm_energy[2] + transition_energy[2]
            ):
                raise RuntimePlanError(
                    "automated cold energy differs from warm plus transition"
                )
        elif any(value is not None for value in energy):
            raise RuntimePlanError(
                "automated fleet energy lacks an explicit decomposition"
            )
        memory = {
            _text("automated memory resource", key): _integer(
                "automated memory bytes", value
            )
            for key, value in self.memory_by_resource_bytes.items()
        }
        object.__setattr__(
            self,
            "memory_by_resource_bytes",
            MappingProxyType(dict(sorted(memory.items()))),
        )
        transfers = tuple(self.transfer_costs)
        if any(not isinstance(row, RuntimeTransferCost) for row in transfers):
            raise RuntimePlanError("automated transfer cost is invalid")
        object.__setattr__(self, "transfer_costs", transfers)
        evidence_levels = {"ABSENT", "ASSUMED", "CALIBRATED", "MEASURED"}
        if (
            self.latency_evidence not in evidence_levels
            or self.energy_evidence not in evidence_levels
        ):
            raise RuntimePlanError("automated cost evidence level is invalid")
        if self.latency_evidence == "ABSENT":
            raise RuntimePlanError("automated latency evidence is absent")
        energy_known = all(value is not None for value in energy)
        if energy_known == (self.energy_evidence == "ABSENT"):
            raise RuntimePlanError(
                "automated energy evidence differs from its bounds"
            )

    def to_json(self) -> dict[str, object]:
        result = {
            "component_energy_uj": self.component_energy_uj,
            "component_service_us": self.component_service_us,
            "compute_us": self.compute_us,
            "energy_evidence": self.energy_evidence,
            "eviction_us": self.eviction_us,
            "exposed_tail_us": self.exposed_tail_us,
            "finish_upper_us": self.finish_upper_us,
            "finish_us": self.finish_us,
            "fleet_energy_lower_uj": self.fleet_energy_lower_uj,
            "fleet_energy_uj": self.fleet_energy_uj,
            "fleet_energy_upper_uj": self.fleet_energy_upper_uj,
            "interference_us": self.interference_us,
            "latency_evidence": self.latency_evidence,
            "join_wait_us": self.join_wait_us,
            "load_us": self.load_us,
            "memory_by_resource_bytes": dict(self.memory_by_resource_bytes),
            "memory_us": self.memory_us,
            "queue_delay_us": self.queue_delay_us,
            "restore_us": self.restore_us,
            "service_upper_us": self.service_upper_us,
            "service_us": self.service_us,
            "start_us": self.start_us,
            "switching_us": self.switching_us,
            "transition_energy_lower_uj": (
                self.transition_energy_lower_uj
            ),
            "transition_energy_uj": self.transition_energy_uj,
            "transition_energy_upper_uj": (
                self.transition_energy_upper_uj
            ),
            "transfer_us": self.transfer_us,
            "transfers": [row.to_json() for row in self.transfer_costs],
            "warm_execution_energy_lower_uj": (
                self.warm_execution_energy_lower_uj
            ),
            "warm_execution_energy_uj": self.warm_execution_energy_uj,
            "warm_execution_energy_upper_uj": (
                self.warm_execution_energy_upper_uj
            ),
        }
        if self.residency_hysteresis_us:
            result["residency_hysteresis_us"] = (
                self.residency_hysteresis_us
            )
        return result
