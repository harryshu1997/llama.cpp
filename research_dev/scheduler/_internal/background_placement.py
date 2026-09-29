"""Periodic preparation of immutable rough placement frontiers."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from types import MappingProxyType
from typing import Callable, Mapping

from .model_manifest import ModelManifest
from .runtime_capabilities import HeterogeneousRuntimeSnapshot
from .runtime_search import RoughPlacementFrontier, request_shape_bucket
from .types import canonical_sha256


class BackgroundPlacementError(ValueError):
    pass


def material_count_bucket(value: int) -> int:
    """Group queue and sample counts by powers of two."""
    if type(value) is not int or value < 0:
        raise BackgroundPlacementError(
            "placement planning count is invalid"
        )
    return 0 if value == 0 else 1 << (value.bit_length() - 1)


def material_horizon_bucket_us(value_us: int) -> int:
    """Group future busy horizons without hiding current availability."""
    if type(value_us) is not int or value_us < 0:
        raise BackgroundPlacementError(
            "placement planning horizon is invalid"
        )
    if value_us == 0:
        return 0
    units = (value_us + 99_999) // 100_000
    return 100_000 * (1 << (units - 1).bit_length())


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise BackgroundPlacementError(f"{name} must be non-empty ASCII text")
    return value


@dataclass(frozen=True)
class PlacementFrontierKey:
    artifact_sha256: str
    input_token_bucket: int
    output_token_bucket: int
    quality_requirement: str
    capability_generation_sha256: str
    residency_generation_sha256: str
    capacity_generation_sha256: str
    cost_profile_generation_sha256: str
    demand_resource_generation_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "artifact_sha256",
            "capability_generation_sha256",
            "residency_generation_sha256",
            "capacity_generation_sha256",
            "cost_profile_generation_sha256",
            "demand_resource_generation_sha256",
        ):
            value = _text("placement frontier " + name, getattr(self, name))
            if not value.startswith("sha256:") or len(value) != 71:
                raise BackgroundPlacementError(
                    "placement frontier hash is invalid"
                )
        for value in (self.input_token_bucket, self.output_token_bucket):
            if type(value) is not int or value <= 0:
                raise BackgroundPlacementError(
                    "placement frontier shape bucket is invalid"
                )
        _text("placement frontier quality", self.quality_requirement)


@dataclass(frozen=True)
class PlacementFrontierEnvelope:
    key: PlacementFrontierKey
    frontier: RoughPlacementFrontier
    planned_at_us: int
    valid_until_us: int
    trigger_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, PlacementFrontierKey):
            raise BackgroundPlacementError(
                "placement frontier key is invalid"
            )
        if not isinstance(self.frontier, RoughPlacementFrontier):
            raise BackgroundPlacementError(
                "placement frontier value is invalid"
            )
        if (
            self.frontier.artifact_sha256 != self.key.artifact_sha256
            or self.frontier.capability_generation_sha256
                != self.key.capability_generation_sha256
            or self.frontier.input_token_bucket
                != self.key.input_token_bucket
            or self.frontier.output_token_bucket
                != self.key.output_token_bucket
        ):
            raise BackgroundPlacementError(
                "placement frontier identity differs"
            )
        if (
            type(self.planned_at_us) is not int
            or self.planned_at_us < 0
            or type(self.valid_until_us) is not int
            or self.valid_until_us <= self.planned_at_us
        ):
            raise BackgroundPlacementError(
                "placement frontier validity is invalid"
            )
        reasons = tuple(sorted(
            _text("placement frontier trigger", value)
            for value in self.trigger_reasons
        ))
        if not reasons or len(reasons) != len(set(reasons)):
            raise BackgroundPlacementError(
                "placement frontier triggers are invalid"
            )
        object.__setattr__(self, "trigger_reasons", reasons)


@dataclass(frozen=True)
class _PlanningDemand:
    key: PlacementFrontierKey
    manifest: ModelManifest
    snapshot: HeterogeneousRuntimeSnapshot
    observed_at_us: int
    trigger_reasons: tuple[str, ...]


FrontierCompiler = Callable[
    [
        ModelManifest,
        int,
        int,
        str,
        HeterogeneousRuntimeSnapshot,
        int,
        tuple[str, ...],
    ],
    RoughPlacementFrontier,
]


def placement_frontier_key(
    *,
    manifest: ModelManifest,
    capability_generation_sha256: str,
    input_tokens: int,
    output_tokens: int,
    quality_requirement: str,
    snapshot: HeterogeneousRuntimeSnapshot,
    online_cost_generation_sha256: str | None = None,
    demand_resource_generation_sha256: str | None = None,
) -> PlacementFrontierKey:
    if not isinstance(manifest, ModelManifest):
        raise BackgroundPlacementError("placement model manifest is invalid")
    if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
        raise BackgroundPlacementError("placement runtime snapshot is invalid")
    input_bucket, output_bucket = request_shape_bucket(
        input_tokens, output_tokens
    )
    if online_cost_generation_sha256 is not None:
        value = _text(
            "online cost generation", online_cost_generation_sha256
        )
        if not value.startswith("sha256:") or len(value) != 71:
            raise BackgroundPlacementError(
                "online cost generation hash is invalid"
            )
    if demand_resource_generation_sha256 is None:
        demand_resource_generation_sha256 = canonical_sha256({
            "scope": "scheduler-supplied-demand-resource-generation",
        })
    else:
        value = _text(
            "demand resource generation",
            demand_resource_generation_sha256,
        )
        if not value.startswith("sha256:") or len(value) != 71:
            raise BackgroundPlacementError(
                "demand resource generation hash is invalid"
            )
    return PlacementFrontierKey(
        artifact_sha256=manifest.artifact_sha256,
        input_token_bucket=input_bucket,
        output_token_bucket=output_bucket,
        quality_requirement=quality_requirement,
        capability_generation_sha256=capability_generation_sha256,
        residency_generation_sha256=canonical_sha256({
            "scope": "request-time-revalidation",
        }),
        capacity_generation_sha256=canonical_sha256({
            "executors": [
                {
                    "executor_id": row.executor_id,
                }
                for row in snapshot.executors.values()
            ],
            "memory": [
                {
                    "capacity_bytes": row.capacity_bytes,
                    "reserve_bytes": row.reserve_bytes,
                    "resource_id": row.resource_id,
                }
                for row in snapshot.memory.capacities.values()
            ],
        }),
        cost_profile_generation_sha256=canonical_sha256({
            "links": [
                {
                    "bandwidth_bytes_per_s": (
                        row.measured_bandwidth_bytes_per_s
                    ),
                    "link_id": row.link_id,
                }
                for row in snapshot.links.values()
            ],
            "online_cost_generation_sha256": (
                online_cost_generation_sha256
            ),
        }),
        demand_resource_generation_sha256=(
            demand_resource_generation_sha256
        ),
    )


class PeriodicPlacementPlanner:
    """Compile rough placement envelopes outside request dispatch threads."""

    def __init__(
        self,
        compile_frontier: FrontierCompiler,
        *,
        refresh_interval_us: int = 30_000_000,
        maximum_entries: int = 512,
        idle_worker_timeout_s: float = 0.05,
    ) -> None:
        if not callable(compile_frontier):
            raise BackgroundPlacementError(
                "placement frontier compiler is invalid"
            )
        if type(refresh_interval_us) is not int or refresh_interval_us <= 0:
            raise BackgroundPlacementError(
                "placement refresh interval is invalid"
            )
        if type(maximum_entries) is not int or maximum_entries <= 0:
            raise BackgroundPlacementError(
                "placement cache capacity is invalid"
            )
        if (
            type(idle_worker_timeout_s) is not float
            or idle_worker_timeout_s <= 0
        ):
            raise BackgroundPlacementError(
                "placement worker idle timeout is invalid"
            )
        self._compile_frontier = compile_frontier
        self._refresh_interval_us = refresh_interval_us
        self._maximum_entries = maximum_entries
        self._idle_worker_timeout_s = idle_worker_timeout_s
        self._condition = threading.Condition()
        self._pending: dict[PlacementFrontierKey, _PlanningDemand] = {}
        self._deferred: dict[PlacementFrontierKey, _PlanningDemand] = {}
        self._active_demands: dict[PlacementFrontierKey, _PlanningDemand] = {}
        self._frontiers: dict[
            PlacementFrontierKey, PlacementFrontierEnvelope
        ] = {}
        self._failures: dict[PlacementFrontierKey, str] = {}
        self._thread: threading.Thread | None = None
        self._stop = False
        self._hits = 0
        self._misses = 0
        self._compilations = 0

    def start(self) -> None:
        with self._condition:
            self._pending.update(self._deferred)
            self._deferred.clear()
            self._start_worker_locked()
            self._condition.notify_all()

    def _start_worker_locked(self) -> None:
        if self._thread is not None:
            return
        if self._stop:
            self._stop = False
        self._thread = threading.Thread(
            target=self._run,
            name="unified-placement-planner",
            daemon=True,
        )
        self._thread.start()

    def close(self, timeout_s: float = 5.0) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                raise BackgroundPlacementError(
                    "placement planner did not stop"
                )
        with self._condition:
            self._thread = None

    def request(
        self,
        *,
        key: PlacementFrontierKey,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        trigger_reasons: tuple[str, ...],
        defer_start: bool = False,
    ) -> None:
        if not isinstance(key, PlacementFrontierKey):
            raise BackgroundPlacementError("placement demand key is invalid")
        if (
            not isinstance(manifest, ModelManifest)
            or manifest.artifact_sha256 != key.artifact_sha256
            or not isinstance(snapshot, HeterogeneousRuntimeSnapshot)
            or type(observed_at_us) is not int
            or observed_at_us < 0
        ):
            raise BackgroundPlacementError("placement demand is invalid")
        snapshot.validate_at(observed_at_us)
        reasons = tuple(sorted(
            _text("placement demand trigger", value)
            for value in trigger_reasons
        ))
        if not reasons:
            raise BackgroundPlacementError(
                "placement demand requires a trigger"
            )
        if type(defer_start) is not bool:
            raise BackgroundPlacementError(
                "placement demand defer flag is invalid"
            )
        demand = _PlanningDemand(
            key=key,
            manifest=manifest,
            snapshot=snapshot,
            observed_at_us=observed_at_us,
            trigger_reasons=reasons,
        )
        with self._condition:
            current = self._frontiers.get(key)
            if current is None or current.valid_until_us <= observed_at_us:
                if defer_start:
                    self._deferred[key] = demand
                else:
                    self._pending[key] = demand
            self._active_demands[key] = demand
            if self._pending and not defer_start:
                self._start_worker_locked()
                self._condition.notify_all()

    def lookup(
        self, key: PlacementFrontierKey, observed_at_us: int
    ) -> PlacementFrontierEnvelope | None:
        if not isinstance(key, PlacementFrontierKey):
            raise BackgroundPlacementError("placement lookup key is invalid")
        if type(observed_at_us) is not int or observed_at_us < 0:
            raise BackgroundPlacementError("placement lookup time is invalid")
        with self._condition:
            envelope = self._frontiers.get(key)
            if envelope is None or envelope.valid_until_us <= observed_at_us:
                self._misses += 1
                return None
            self._hits += 1
            return envelope

    def refresh_one(self, key: PlacementFrontierKey) -> None:
        with self._condition:
            demand = self._active_demands.get(key)
        if demand is None:
            raise BackgroundPlacementError("placement demand is absent")
        self._compile(demand)

    def _compile(self, demand: _PlanningDemand) -> None:
        try:
            frontier = self._compile_frontier(
                demand.manifest,
                demand.key.input_token_bucket,
                demand.key.output_token_bucket,
                demand.key.quality_requirement,
                demand.snapshot,
                demand.observed_at_us,
                demand.trigger_reasons,
            )
            envelope = PlacementFrontierEnvelope(
                key=demand.key,
                frontier=frontier,
                planned_at_us=demand.observed_at_us,
                valid_until_us=min(
                    demand.snapshot.valid_until_us,
                    demand.observed_at_us + self._refresh_interval_us,
                ),
                trigger_reasons=demand.trigger_reasons,
            )
        except BaseException as exc:
            with self._condition:
                self._failures[demand.key] = (
                    type(exc).__name__ + ": " + str(exc)
                )
                self._condition.notify_all()
            return
        with self._condition:
            if len(self._frontiers) >= self._maximum_entries:
                oldest = min(
                    self._frontiers,
                    key=lambda key: (
                        self._frontiers[key].planned_at_us,
                        repr(key),
                    ),
                )
                del self._frontiers[oldest]
            self._frontiers[demand.key] = envelope
            self._failures.pop(demand.key, None)
            self._compilations += 1
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._stop:
                    self._thread = None
                    return
                if not self._pending:
                    self._condition.wait(self._idle_worker_timeout_s)
                    if self._stop:
                        self._thread = None
                        return
                    if not self._pending:
                        self._thread = None
                        return
                key = min(self._pending, key=repr)
                demand = self._pending.pop(key)
            self._compile(demand)

    def stats(self) -> Mapping[str, int]:
        with self._condition:
            return MappingProxyType({
                "active_demands": len(self._active_demands),
                "compilations": self._compilations,
                "entries": len(self._frontiers),
                "failures": len(self._failures),
                "hits": self._hits,
                "misses": self._misses,
                "pending": len(self._pending) + len(self._deferred),
            })
