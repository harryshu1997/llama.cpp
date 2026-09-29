"""Runtime observations and scheduler-owned executor admission."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from .policy import ProfileBundle, Request
from .runtime_cost import (
    RuntimeExecutorBinding,
    RuntimeMemoryDemand,
    RuntimeModelArtifact,
)


class RuntimeAdmissionError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise RuntimeAdmissionError(f"{name} must be non-empty ASCII text")
    return value


def _integer(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeAdmissionError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class RuntimeRequestObservation:
    captured_at_us: int
    cost_features: Mapping[str, int]
    memory_occupied_bytes: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _integer("runtime observation captured_at_us", self.captured_at_us)
        features = {
            _text("runtime cost feature", name): _integer(
                f"runtime cost feature {name}", value
            )
            for name, value in self.cost_features.items()
        }
        object.__setattr__(
            self,
            "cost_features",
            MappingProxyType(dict(sorted(features.items()))),
        )
        memory = {
            _text("runtime memory resource", name): _integer(
                f"runtime occupied memory {name}", value
            )
            for name, value in self.memory_occupied_bytes.items()
        }
        object.__setattr__(
            self,
            "memory_occupied_bytes",
            MappingProxyType(dict(sorted(memory.items()))),
        )

    def evaluation_request(self, request: Request) -> Request:
        if not isinstance(request, Request):
            raise RuntimeAdmissionError("runtime request is invalid")
        result = replace(request, features=self.cost_features)
        result.validate()
        return result

    def to_json(self) -> dict[str, object]:
        result = {
            "captured_at_us": self.captured_at_us,
            "cost_features": dict(self.cost_features),
        }
        if self.memory_occupied_bytes:
            result["memory_occupied_bytes"] = dict(
                self.memory_occupied_bytes
            )
        return result


@dataclass(frozen=True)
class RuntimeExecutorObservation:
    executor_id: str
    route_id: str
    backend: str
    resource_ids: tuple[str, ...]
    memory_resource_id: str | None
    resident: bool
    health: str
    qualification_facts: Mapping[str, bool]
    memory_demands: tuple[RuntimeMemoryDemand, ...] = ()
    route_family: str | None = None

    def __post_init__(self) -> None:
        for name in ("executor_id", "route_id", "backend", "health"):
            _text(f"runtime executor observation {name}", getattr(self, name))
        resources = tuple(
            _text("runtime executor observation resource", value)
            for value in self.resource_ids
        )
        if not resources or len(resources) != len(set(resources)):
            raise RuntimeAdmissionError(
                "runtime executor observation resources must be unique"
            )
        if self.memory_resource_id is not None:
            _text(
                "runtime executor observation memory resource",
                self.memory_resource_id,
            )
        if type(self.resident) is not bool:
            raise RuntimeAdmissionError(
                "runtime executor observation resident must be bool"
            )
        facts = {}
        for name, value in self.qualification_facts.items():
            name = _text("runtime qualification fact", name)
            if type(value) is not bool:
                raise RuntimeAdmissionError(
                    f"runtime qualification fact {name} must be bool"
                )
            facts[name] = value
        demands = tuple(self.memory_demands)
        if any(not isinstance(row, RuntimeMemoryDemand) for row in demands):
            raise RuntimeAdmissionError(
                "runtime executor observation memory demand is invalid"
            )
        if self.route_family is not None:
            _text(
                "runtime executor observation route_family",
                self.route_family,
            )
        object.__setattr__(self, "resource_ids", resources)
        object.__setattr__(
            self,
            "qualification_facts",
            MappingProxyType(dict(sorted(facts.items()))),
        )
        object.__setattr__(self, "memory_demands", demands)

    @property
    def eligibility_reasons(self) -> tuple[str, ...]:
        reasons = []
        if self.health != "healthy":
            reasons.append("HEALTH:" + self.health)
        reasons.extend(
            "QUALIFICATION:" + name
            for name, value in self.qualification_facts.items()
            if not value
        )
        return tuple(reasons)

    def to_json(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "eligibility_reasons": list(self.eligibility_reasons),
            "executor_id": self.executor_id,
            "health": self.health,
            "memory_demands": [row.to_json() for row in self.memory_demands],
            "memory_resource_id": self.memory_resource_id,
            "qualification_facts": dict(self.qualification_facts),
            "resident": self.resident,
            "resource_ids": list(self.resource_ids),
            "route_family": self.route_family,
            "route_id": self.route_id,
        }


class RuntimeCandidateBuilder:
    """Build bindings from raw executor observations and route profiles."""

    @staticmethod
    def build(
        profile: ProfileBundle,
        model: RuntimeModelArtifact,
        observations: Sequence[RuntimeExecutorObservation],
    ) -> tuple[RuntimeExecutorBinding, ...]:
        if not isinstance(profile, ProfileBundle):
            raise RuntimeAdmissionError("runtime profile is invalid")
        if not isinstance(model, RuntimeModelArtifact):
            raise RuntimeAdmissionError("runtime model is invalid")
        rows = tuple(observations)
        if not rows or any(
            not isinstance(row, RuntimeExecutorObservation) for row in rows
        ):
            raise RuntimeAdmissionError(
                "runtime executor observations are invalid"
            )
        route_ids = {route.route_id for route in profile.routes}
        observed_ids = [row.route_id for row in rows]
        if len(observed_ids) != len(set(observed_ids)):
            raise RuntimeAdmissionError(
                "runtime executor observation route ids are not unique"
            )
        unknown = sorted(set(observed_ids) - route_ids)
        if unknown:
            raise RuntimeAdmissionError(
                "runtime executor observation references unknown route: "
                + ", ".join(unknown)
            )
        return tuple(
            RuntimeExecutorBinding(
                executor_id=row.executor_id,
                route_id=row.route_id,
                model_id=model.model_id,
                artifact_sha256=model.artifact_sha256,
                artifact_bytes=model.artifact_bytes,
                backend=row.backend,
                resource_ids=row.resource_ids,
                memory_resource_id=row.memory_resource_id,
                resident=row.resident,
                ready=not row.eligibility_reasons,
                memory_demands=row.memory_demands,
                route_family=row.route_family,
                eligibility_reasons=row.eligibility_reasons,
            )
            for row in rows
        )
