"""Shape-aware route cost estimates bound to live executor residency."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from .policy import ProfileBundle, Request, RouteProfile, SchedulerError
from .runtime_placement import RuntimePlacementSnapshot


RUNTIME_COST_SCHEMA = "research-scheduler-runtime-cost-v3"

__all__ = [
    "RUNTIME_COST_SCHEMA",
    "RuntimeCostError",
    "RuntimeCostEstimateSet",
    "RuntimeCostEstimator",
    "RuntimeExecutorBinding",
    "RuntimeExecutorRegistry",
    "RuntimeMemoryDemand",
    "RuntimeModelArtifact",
    "RuntimeRouteCostEstimate",
]


class RuntimeCostError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise RuntimeCostError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeCostError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeCostError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise RuntimeCostError(f"{name} must be bool")
    return value


def _sha256(name: str, value: object) -> str:
    digest = _text(name, value).removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise RuntimeCostError(f"{name} must be a lowercase SHA-256")
    return "sha256:" + digest


def _unique_texts(name: str, values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(_text(name, value) for value in values)
    if not result or len(result) != len(set(result)):
        raise RuntimeCostError(f"{name} must be non-empty and unique")
    return result


RUNTIME_MEMORY_LIFETIMES = frozenset({"request", "resident"})


@dataclass(frozen=True)
class RuntimeMemoryDemand:
    demand_id: str
    resource_id: str
    kind: str
    required_bytes: int
    resident_bytes: int
    lifetime: str

    def __post_init__(self) -> None:
        for name in ("demand_id", "resource_id", "kind"):
            _text(f"runtime memory {name}", getattr(self, name))
        _integer("runtime memory required_bytes", self.required_bytes, 1)
        _integer("runtime memory resident_bytes", self.resident_bytes)
        if self.resident_bytes > self.required_bytes:
            raise RuntimeCostError(
                "runtime memory resident bytes exceed required bytes"
            )
        if self.lifetime not in RUNTIME_MEMORY_LIFETIMES:
            raise RuntimeCostError("runtime memory lifetime is invalid")

    @property
    def additional_bytes(self) -> int:
        return self.required_bytes - self.resident_bytes

    @property
    def residency_satisfied(self) -> bool:
        return (
            self.lifetime == "request"
            or self.additional_bytes == 0
        )

    def to_json(self) -> dict[str, int | str]:
        return {
            "additional_bytes": self.additional_bytes,
            "demand_id": self.demand_id,
            "kind": self.kind,
            "lifetime": self.lifetime,
            "required_bytes": self.required_bytes,
            "resident_bytes": self.resident_bytes,
            "resource_id": self.resource_id,
        }


@dataclass(frozen=True)
class RuntimeModelArtifact:
    model_id: str
    artifact_sha256: str
    artifact_bytes: int

    def __post_init__(self) -> None:
        _text("runtime model id", self.model_id)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256("runtime model artifact_sha256", self.artifact_sha256),
        )
        _integer("runtime model artifact_bytes", self.artifact_bytes, 1)

    def to_json(self) -> dict[str, int | str]:
        return {
            "artifact_bytes": self.artifact_bytes,
            "artifact_sha256": self.artifact_sha256,
            "model_id": self.model_id,
        }


@dataclass(frozen=True)
class RuntimeExecutorBinding:
    executor_id: str
    route_id: str
    model_id: str
    artifact_sha256: str
    artifact_bytes: int
    backend: str
    resource_ids: tuple[str, ...]
    memory_resource_id: str | None
    resident: bool
    ready: bool
    memory_demands: tuple[RuntimeMemoryDemand, ...] = ()
    queueable: bool = False
    residency_candidate_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "executor_id",
            "route_id",
            "model_id",
            "backend",
        ):
            _text(f"runtime executor {name}", getattr(self, name))
        if self.memory_resource_id is not None:
            _text(
                "runtime executor memory_resource_id",
                self.memory_resource_id,
            )
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(
                "runtime executor artifact_sha256", self.artifact_sha256
            ),
        )
        _integer("runtime executor artifact_bytes", self.artifact_bytes, 1)
        object.__setattr__(
            self,
            "resource_ids",
            _unique_texts("runtime executor resource id", self.resource_ids),
        )
        _boolean("runtime executor resident", self.resident)
        _boolean("runtime executor ready", self.ready)
        _boolean("runtime executor queueable", self.queueable)
        if self.residency_candidate_id is not None:
            _text(
                "runtime executor residency_candidate_id",
                self.residency_candidate_id,
            )
        demands = tuple(self.memory_demands)
        if not demands:
            if self.memory_resource_id is None:
                raise RuntimeCostError(
                    "runtime executor requires memory demands"
                )
            demands = (RuntimeMemoryDemand(
                demand_id="model-weights",
                resource_id=self.memory_resource_id,
                kind="model_weights",
                required_bytes=self.artifact_bytes,
                resident_bytes=(self.artifact_bytes if self.resident else 0),
                lifetime="resident",
            ),)
        if (
            any(not isinstance(item, RuntimeMemoryDemand) for item in demands)
            or len({item.demand_id for item in demands}) != len(demands)
        ):
            raise RuntimeCostError(
                "runtime executor memory demands are invalid"
            )
        persistent_resident = all(
            demand.residency_satisfied for demand in demands
        )
        if self.resident != persistent_resident:
            raise RuntimeCostError(
                "runtime executor residency differs from memory demands"
            )
        object.__setattr__(
            self,
            "memory_demands",
            tuple(sorted(demands, key=lambda item: item.demand_id)),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "artifact_bytes": self.artifact_bytes,
            "artifact_sha256": self.artifact_sha256,
            "backend": self.backend,
            "executor_id": self.executor_id,
            "memory_resource_id": self.memory_resource_id,
            "memory_demands": [
                demand.to_json() for demand in self.memory_demands
            ],
            "model_id": self.model_id,
            "ready": self.ready,
            "resident": self.resident,
            "queueable": self.queueable,
            "residency_candidate_id": self.residency_candidate_id,
            "resource_ids": list(self.resource_ids),
            "route_id": self.route_id,
        }


@dataclass(frozen=True)
class RuntimeExecutorRegistry:
    bindings: Mapping[str, RuntimeExecutorBinding]

    def __post_init__(self) -> None:
        rows = dict(self.bindings)
        if not rows:
            raise RuntimeCostError("runtime executor registry is empty")
        if any(
            not isinstance(binding, RuntimeExecutorBinding)
            or route_id != binding.route_id
            for route_id, binding in rows.items()
        ):
            raise RuntimeCostError(
                "runtime executor registry bindings are invalid"
            )
        object.__setattr__(
            self,
            "bindings",
            MappingProxyType(dict(sorted(rows.items()))),
        )

    @classmethod
    def from_bindings(
        cls, bindings: Sequence[RuntimeExecutorBinding]
    ) -> "RuntimeExecutorRegistry":
        rows = tuple(bindings)
        if any(not isinstance(row, RuntimeExecutorBinding) for row in rows):
            raise RuntimeCostError("runtime executor binding is invalid")
        by_route = {row.route_id: row for row in rows}
        if len(by_route) != len(rows):
            raise RuntimeCostError(
                "runtime executor registry route ids are not unique"
            )
        return cls(by_route)

    def binding(self, route_id: str) -> RuntimeExecutorBinding | None:
        return self.bindings.get(_text("runtime registry route_id", route_id))

    def to_json(self) -> dict[str, object]:
        return {
            "bindings": [
                binding.to_json() for binding in self.bindings.values()
            ],
            "schema": "research-scheduler-executor-registry-v1",
        }


@dataclass(frozen=True)
class RuntimeRouteCostEstimate:
    route_id: str
    executor_id: str | None
    baseline: bool
    admitted: bool
    reason: str
    service_us: int
    service_upper_us: int
    latency_profile_label: str
    latency_sample_count: int
    latency_measured: bool
    fleet_energy_uj: int | None
    fleet_energy_lower_uj: int | None
    fleet_energy_upper_uj: int | None
    additional_bytes: int
    memory_resource_id: str | None
    additional_bytes_by_resource: Mapping[str, int]
    memory_demands: tuple[RuntimeMemoryDemand, ...]

    def __post_init__(self) -> None:
        _text("runtime cost route_id", self.route_id)
        if self.executor_id is not None:
            _text("runtime cost executor_id", self.executor_id)
        _boolean("runtime cost baseline", self.baseline)
        _boolean("runtime cost admitted", self.admitted)
        _text("runtime cost reason", self.reason)
        _integer("runtime cost service_us", self.service_us, 1)
        _integer(
            "runtime cost service_upper_us",
            self.service_upper_us,
            self.service_us,
        )
        _text(
            "runtime cost latency_profile_label",
            self.latency_profile_label,
        )
        _integer(
            "runtime cost latency_sample_count",
            self.latency_sample_count,
            1,
        )
        _boolean("runtime cost latency_measured", self.latency_measured)
        for name in (
            "fleet_energy_uj",
            "fleet_energy_lower_uj",
            "fleet_energy_upper_uj",
        ):
            value = getattr(self, name)
            if value is not None:
                _integer(f"runtime cost {name}", value, 1)
        _integer("runtime cost additional_bytes", self.additional_bytes)
        if self.memory_resource_id is not None:
            _text(
                "runtime cost memory_resource_id", self.memory_resource_id
            )
        additional = {
            _text("runtime cost memory resource", resource_id): _integer(
                "runtime cost additional resource bytes", amount
            )
            for resource_id, amount in self.additional_bytes_by_resource.items()
        }
        if sum(additional.values()) != self.additional_bytes:
            raise RuntimeCostError(
                "runtime cost additional memory total differs"
            )
        demands = tuple(self.memory_demands)
        if any(not isinstance(item, RuntimeMemoryDemand) for item in demands):
            raise RuntimeCostError("runtime cost memory demands are invalid")
        object.__setattr__(
            self,
            "additional_bytes_by_resource",
            MappingProxyType(dict(sorted(additional.items()))),
        )
        object.__setattr__(self, "memory_demands", demands)

    def to_json(self) -> dict[str, object]:
        return {
            "additional_bytes": self.additional_bytes,
            "additional_bytes_by_resource": dict(
                self.additional_bytes_by_resource
            ),
            "admitted": self.admitted,
            "baseline": self.baseline,
            "executor_id": self.executor_id,
            "fleet_energy_lower_uj": self.fleet_energy_lower_uj,
            "fleet_energy_uj": self.fleet_energy_uj,
            "fleet_energy_upper_uj": self.fleet_energy_upper_uj,
            "memory_resource_id": self.memory_resource_id,
            "memory_demands": [
                demand.to_json() for demand in self.memory_demands
            ],
            "latency_measured": self.latency_measured,
            "latency_profile_label": self.latency_profile_label,
            "latency_sample_count": self.latency_sample_count,
            "reason": self.reason,
            "route_id": self.route_id,
            "service_upper_us": self.service_upper_us,
            "service_us": self.service_us,
        }


@dataclass(frozen=True)
class RuntimeCostEstimateSet:
    request_id: str
    workload_id: str
    model: RuntimeModelArtifact
    snapshot: RuntimePlacementSnapshot
    baseline_route_id: str
    estimates: tuple[RuntimeRouteCostEstimate, ...]

    def __post_init__(self) -> None:
        _text("runtime cost request_id", self.request_id)
        _text("runtime cost workload_id", self.workload_id)
        if not isinstance(self.model, RuntimeModelArtifact):
            raise RuntimeCostError("runtime cost model is invalid")
        if not isinstance(self.snapshot, RuntimePlacementSnapshot):
            raise RuntimeCostError("runtime cost snapshot is invalid")
        _text("runtime cost baseline_route_id", self.baseline_route_id)
        estimates = tuple(self.estimates)
        if not estimates or any(
            not isinstance(item, RuntimeRouteCostEstimate)
            for item in estimates
        ):
            raise RuntimeCostError("runtime cost estimates are invalid")
        ids = [item.route_id for item in estimates]
        if len(ids) != len(set(ids)):
            raise RuntimeCostError("runtime cost route ids are not unique")
        baseline = next(
            (
                item for item in estimates
                if item.route_id == self.baseline_route_id
            ),
            None,
        )
        if baseline is None or not baseline.baseline or not baseline.admitted:
            raise RuntimeCostError(
                "runtime cost requires an admitted baseline fallback"
            )
        object.__setattr__(self, "estimates", tuple(sorted(
            estimates, key=lambda item: item.route_id
        )))

    def to_json(self) -> dict[str, object]:
        return {
            "baseline_route_id": self.baseline_route_id,
            "estimates": [item.to_json() for item in self.estimates],
            "model": self.model.to_json(),
            "request_id": self.request_id,
            "schema": RUNTIME_COST_SCHEMA,
            "snapshot": self.snapshot.to_json(),
            "workload_id": self.workload_id,
        }


class RuntimeCostEstimator:
    @staticmethod
    def _reason(
        route: RouteProfile,
        binding: RuntimeExecutorBinding | None,
        model: RuntimeModelArtifact,
        snapshot: RuntimePlacementSnapshot,
        request: Request,
    ) -> tuple[
        str | None,
        int,
        str | None,
        Mapping[str, int],
        tuple[RuntimeMemoryDemand, ...],
    ]:
        if binding is None:
            return "EXECUTOR_ABSENT", 0, None, {}, ()
        if binding.route_id != route.route_id:
            return (
                "ROUTE_BINDING_MISMATCH", 0, binding.memory_resource_id,
                {}, binding.memory_demands,
            )
        if binding.model_id != model.model_id:
            return (
                "MODEL_ID_MISMATCH", 0, binding.memory_resource_id,
                {}, binding.memory_demands,
            )
        if binding.artifact_sha256 != model.artifact_sha256:
            return (
                "MODEL_HASH_MISMATCH", 0, binding.memory_resource_id,
                {}, binding.memory_demands,
            )
        if binding.artifact_bytes != model.artifact_bytes:
            return (
                "MODEL_BYTES_MISMATCH", 0, binding.memory_resource_id,
                {}, binding.memory_demands,
            )
        if set(binding.resource_ids) != set(route.resource_slots):
            return (
                "RESOURCE_BINDING_MISMATCH", 0,
                binding.memory_resource_id, {}, binding.memory_demands,
            )
        additional_by_resource: dict[str, int] = {}
        required_by_resource: dict[str, int] = {}
        for demand in binding.memory_demands:
            required_by_resource[demand.resource_id] = (
                required_by_resource.get(demand.resource_id, 0)
                + demand.required_bytes
            )
            additional_by_resource[demand.resource_id] = (
                additional_by_resource.get(demand.resource_id, 0)
                + demand.additional_bytes
            )
        additional_bytes = sum(additional_by_resource.values())
        for resource_id, required_bytes in sorted(
            required_by_resource.items()
        ):
            capacity = snapshot.capacities.get(resource_id)
            if capacity is None:
                return (
                    "MEMORY_RESOURCE_ABSENT", additional_bytes,
                    resource_id, additional_by_resource,
                    binding.memory_demands,
                )
            if required_bytes > (
                capacity.capacity_bytes - capacity.reserve_bytes
            ):
                return (
                    "MEMORY_DEMAND_DOES_NOT_FIT", additional_bytes,
                    resource_id, additional_by_resource,
                    binding.memory_demands,
                )
            if additional_by_resource[resource_id] > capacity.available_bytes:
                return (
                    "CAPACITY", additional_bytes, resource_id,
                    additional_by_resource, binding.memory_demands,
                )
        if not binding.ready and not binding.queueable:
            return (
                "EXECUTOR_NOT_READY", additional_bytes,
                binding.memory_resource_id, additional_by_resource,
                binding.memory_demands,
            )
        if not binding.resident:
            return (
                "WEIGHTS_NOT_RESIDENT", additional_bytes,
                binding.memory_resource_id, additional_by_resource,
                binding.memory_demands,
            )
        if not route.placement_verified:
            return (
                "PLACEMENT_UNVERIFIED", additional_bytes,
                binding.memory_resource_id, additional_by_resource,
                binding.memory_demands,
            )
        if not route.resident:
            return (
                "PROFILE_NOT_RESIDENT", additional_bytes,
                binding.memory_resource_id, additional_by_resource,
                binding.memory_demands,
            )
        if not route.latency.measured_for(request):
            return (
                "LATENCY_UNMEASURED", additional_bytes,
                binding.memory_resource_id, additional_by_resource,
                binding.memory_demands,
            )
        return (
            None, additional_bytes, binding.memory_resource_id,
            additional_by_resource, binding.memory_demands,
        )

    def estimate(
        self,
        *,
        profile: ProfileBundle,
        request: Request,
        model: RuntimeModelArtifact,
        bindings: Sequence[RuntimeExecutorBinding],
        snapshot: RuntimePlacementSnapshot,
        now_us: int,
    ) -> RuntimeCostEstimateSet:
        if not isinstance(profile, ProfileBundle):
            raise RuntimeCostError("runtime cost profile is invalid")
        if not isinstance(request, Request):
            raise RuntimeCostError("runtime cost request is invalid")
        try:
            request.validate()
        except SchedulerError as exc:
            raise RuntimeCostError(str(exc)) from exc
        if not isinstance(model, RuntimeModelArtifact):
            raise RuntimeCostError("runtime cost model is invalid")
        if not isinstance(snapshot, RuntimePlacementSnapshot):
            raise RuntimeCostError("runtime cost snapshot is invalid")
        _integer("runtime cost now_us", now_us)
        if not snapshot.captured_at_us <= now_us < snapshot.valid_until_us:
            raise RuntimeCostError("runtime cost snapshot is stale")
        mapped_workload = profile.trace_workload_map.get(model.model_id)
        if mapped_workload != request.workload_id:
            raise RuntimeCostError(
                "runtime model does not map to the request workload"
            )
        binding_rows = tuple(bindings)
        if any(
            not isinstance(item, RuntimeExecutorBinding)
            for item in binding_rows
        ):
            raise RuntimeCostError("runtime executor binding is invalid")
        by_route = {item.route_id: item for item in binding_rows}
        if len(by_route) != len(binding_rows):
            raise RuntimeCostError("runtime executor route ids are not unique")
        routes = tuple(
            route for route in profile.routes
            if route.workload_id == request.workload_id
        )
        if not routes:
            raise RuntimeCostError("runtime request has no route profiles")
        baseline = next((route for route in routes if route.baseline), None)
        if baseline is None:
            raise RuntimeCostError("runtime request has no baseline route")

        estimates = []
        for route in routes:
            binding = by_route.get(route.route_id)
            try:
                variant = route.latency.variant(request)
                service_us = route.latency.predict_us(request)
                service_upper_us = route.latency.upper_us(request)
                energy, lower, upper, _ = route.energy.bounds_uj(
                    request, service_us
                )
            except SchedulerError as exc:
                raise RuntimeCostError(
                    f"runtime route cost failed for {route.route_id}: {exc}"
                ) from exc
            (
                reason,
                additional_bytes,
                memory_resource_id,
                additional_bytes_by_resource,
                memory_demands,
            ) = self._reason(route, binding, model, snapshot, request)
            estimates.append(RuntimeRouteCostEstimate(
                route_id=route.route_id,
                executor_id=(
                    None if binding is None else binding.executor_id
                ),
                baseline=route.baseline,
                admitted=reason is None,
                reason="ADMITTED" if reason is None else reason,
                service_us=service_us,
                service_upper_us=service_upper_us,
                latency_profile_label=(
                    "default" if variant is None else variant.label
                ),
                latency_sample_count=(
                    route.latency.sample_count
                    if variant is None
                    else variant.sample_count
                ),
                latency_measured=(
                    route.latency.measured
                    if variant is None
                    else variant.measured
                ),
                fleet_energy_uj=energy,
                fleet_energy_lower_uj=lower,
                fleet_energy_upper_uj=upper,
                additional_bytes=additional_bytes,
                memory_resource_id=memory_resource_id,
                additional_bytes_by_resource=additional_bytes_by_resource,
                memory_demands=memory_demands,
            ))
        try:
            return RuntimeCostEstimateSet(
                request_id=request.request_id,
                workload_id=request.workload_id,
                model=model,
                snapshot=snapshot,
                baseline_route_id=baseline.route_id,
                estimates=tuple(estimates),
            )
        except RuntimeCostError as exc:
            baseline_estimate = next(
                item for item in estimates if item.baseline
            )
            raise RuntimeCostError(
                "runtime baseline fallback is unavailable: "
                + baseline_estimate.reason
            ) from exc
