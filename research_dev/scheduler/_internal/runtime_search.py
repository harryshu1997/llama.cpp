"""Deterministic bounded search over rough heterogeneous placements."""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
from types import MappingProxyType
from typing import Mapping, Sequence


class RuntimeSearchError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise RuntimeSearchError(f"{name} must be non-empty ASCII text")
    return value


def request_shape_bucket(input_tokens: int, output_tokens: int) -> tuple[int, int]:
    """Return conservative power-of-two buckets for a request shape."""
    if type(input_tokens) is not int or input_tokens <= 0:
        raise RuntimeSearchError("input tokens must be positive")
    if type(output_tokens) is not int or output_tokens <= 0:
        raise RuntimeSearchError("output tokens must be positive")

    def upper_power_of_two(value: int) -> int:
        return 1 << (value - 1).bit_length()

    return upper_power_of_two(input_tokens), upper_power_of_two(output_tokens)


@dataclass(frozen=True)
class RoughPlacementVisit:
    route_key: str
    residency_variant: str
    rough_latency_us: int
    rough_energy_uj: int
    required_group: str | None = None
    coverage_group: str | None = None
    coverage_only: bool = False
    rough_memory_feasible: bool = True
    mandatory: bool = False

    def __post_init__(self) -> None:
        _text("rough route key", self.route_key)
        _text("rough residency variant", self.residency_variant)
        if type(self.rough_latency_us) is not int or self.rough_latency_us <= 0:
            raise RuntimeSearchError("rough latency must be positive")
        if type(self.rough_energy_uj) is not int or self.rough_energy_uj <= 0:
            raise RuntimeSearchError("rough energy must be positive")
        if self.required_group is not None:
            _text("rough required group", self.required_group)
        if self.coverage_group is not None:
            _text("rough coverage group", self.coverage_group)
        if type(self.coverage_only) is not bool:
            raise RuntimeSearchError("rough coverage-only flag is invalid")
        if self.coverage_only and self.coverage_group is None:
            raise RuntimeSearchError(
                "rough coverage-only visit requires a coverage group"
            )
        if type(self.rough_memory_feasible) is not bool:
            raise RuntimeSearchError(
                "rough memory feasibility must be boolean"
            )
        if type(self.mandatory) is not bool:
            raise RuntimeSearchError(
                "rough mandatory flag must be boolean"
            )

    @property
    def visit_id(self) -> str:
        return self.route_key + ":residency:" + self.residency_variant


@dataclass(frozen=True)
class RoughPlacementFrontier:
    artifact_sha256: str
    capability_generation_sha256: str
    input_token_bucket: int
    output_token_bucket: int
    search_budget: int
    visits: tuple[RoughPlacementVisit, ...]
    rough_plan_count: int

    def __post_init__(self) -> None:
        for name in ("artifact_sha256", "capability_generation_sha256"):
            value = _text("rough frontier " + name, getattr(self, name))
            if not value.startswith("sha256:") or len(value) != 71:
                raise RuntimeSearchError("rough frontier hash is invalid")
        for value in (self.input_token_bucket, self.output_token_bucket):
            if type(value) is not int or value <= 0:
                raise RuntimeSearchError("rough frontier bucket is invalid")
        if type(self.search_budget) is not int or not 1 <= self.search_budget <= 32:
            raise RuntimeSearchError("rough frontier budget is invalid")
        rows = tuple(self.visits)
        if not rows or len(rows) > self.search_budget:
            raise RuntimeSearchError("rough frontier visits exceed the budget")
        if len({row.visit_id for row in rows}) != len(rows):
            raise RuntimeSearchError("rough frontier visits are duplicated")
        if type(self.rough_plan_count) is not int or self.rough_plan_count < len(rows):
            raise RuntimeSearchError("rough plan count is invalid")
        object.__setattr__(self, "visits", rows)

    def metadata(self, *, cache_hit: bool) -> Mapping[str, object]:
        return MappingProxyType({
            "cache_hit": cache_hit,
            "capability_generation_sha256": self.capability_generation_sha256,
            "evaluated_plan_count": len(self.visits),
            "input_token_bucket": self.input_token_bucket,
            "output_token_bucket": self.output_token_bucket,
            "rough_plan_count": self.rough_plan_count,
            "search_budget": self.search_budget,
            "search_kind": "bounded-edit-refinement-v1",
            "visited_plan_ids": tuple(row.visit_id for row in self.visits),
        })


class BoundedPlacementCompiler:
    """Cache a rough frontier and cap expensive online evaluation at 32."""

    def __init__(
        self,
        search_budget: int = 32,
        refinement_budget: int | None = None,
        maximum_entries: int = 512,
    ) -> None:
        if type(search_budget) is not int or not 1 <= search_budget <= 32:
            raise RuntimeSearchError("runtime search budget is invalid")
        refinement_budget = (
            search_budget
            if refinement_budget is None
            else refinement_budget
        )
        if (
            type(refinement_budget) is not int
            or not 1 <= refinement_budget <= search_budget
        ):
            raise RuntimeSearchError(
                "runtime refinement budget is invalid"
            )
        if type(maximum_entries) is not int or maximum_entries <= 0:
            raise RuntimeSearchError(
                "runtime search cache limit is invalid"
            )
        self.search_budget = search_budget
        self.refinement_budget = refinement_budget
        self.maximum_entries = maximum_entries
        self._cache: dict[tuple[object, ...], RoughPlacementFrontier] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.generations = 0

    @staticmethod
    def _rank(row: RoughPlacementVisit) -> tuple[object, ...]:
        return (
            not row.rough_memory_feasible,
            row.rough_energy_uj,
            row.rough_latency_us,
            row.visit_id,
        )

    @staticmethod
    def _key(
        artifact_sha256: str,
        capability_generation_sha256: str,
        input_tokens: int,
        output_tokens: int,
        quality_requirement: str,
    ) -> tuple[object, ...]:
        input_bucket, output_bucket = request_shape_bucket(
            input_tokens, output_tokens
        )
        return (
            artifact_sha256,
            capability_generation_sha256,
            input_bucket,
            output_bucket,
            quality_requirement,
        )

    def lookup(
        self,
        *,
        artifact_sha256: str,
        capability_generation_sha256: str,
        input_tokens: int,
        output_tokens: int,
        quality_requirement: str,
    ) -> RoughPlacementFrontier | None:
        key = self._key(
            artifact_sha256,
            capability_generation_sha256,
            input_tokens,
            output_tokens,
            quality_requirement,
        )
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self.hits += 1
            return cached

    def compile(
        self,
        *,
        artifact_sha256: str,
        capability_generation_sha256: str,
        input_tokens: int,
        output_tokens: int,
        quality_requirement: str,
        visits: Sequence[RoughPlacementVisit],
        replace_existing: bool = False,
    ) -> tuple[RoughPlacementFrontier, bool]:
        input_bucket, output_bucket = request_shape_bucket(input_tokens, output_tokens)
        key = self._key(
            artifact_sha256,
            capability_generation_sha256,
            input_tokens,
            output_tokens,
            quality_requirement,
        )
        if type(replace_existing) is not bool:
            raise RuntimeSearchError(
                "rough placement replacement flag is invalid"
            )
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and not replace_existing:
                self.hits += 1
                return cached, True
            rows = tuple(visits)
            if not rows:
                raise RuntimeSearchError(
                    "rough placement compiler has no plans"
                )
            by_id = {row.visit_id: row for row in rows}
            if len(by_id) != len(rows):
                raise RuntimeSearchError(
                    "rough placement plans are duplicated"
                )

            mandatory = sorted(
                (row for row in rows if row.mandatory),
                key=self._rank,
            )
            if len(mandatory) > self.refinement_budget:
                raise RuntimeSearchError(
                    "mandatory rough placements exceed the refinement budget"
                )
            selected_ids = {row.visit_id for row in mandatory}
            required = []
            by_group: dict[str, list[RoughPlacementVisit]] = {}
            for row in rows:
                if (
                    row.visit_id not in selected_ids
                    and row.required_group is not None
                ):
                    by_group.setdefault(row.required_group, []).append(row)
            for group in sorted(by_group):
                required.append(min(by_group[group], key=self._rank))
            remaining_budget = self.refinement_budget - len(mandatory)
            if len(required) > remaining_budget:
                required = sorted(required, key=self._rank)[
                    :remaining_budget
                ]
            selected_ids.update(row.visit_id for row in required)
            remaining = sorted(
                (row for row in rows if row.visit_id not in selected_ids),
                key=self._rank,
            )
            selected = mandatory + required + remaining[
                :self.refinement_budget - len(mandatory) - len(required)
            ]
            selected_ids = {row.visit_id for row in selected}
            observed_coverage = {
                row.coverage_group for row in selected
                if row.coverage_group is not None
            }
            by_coverage: dict[str, list[RoughPlacementVisit]] = {}
            for row in rows:
                if row.coverage_group is not None:
                    by_coverage.setdefault(row.coverage_group, []).append(row)
            missing_coverage = tuple(
                group for group in sorted(by_coverage)
                if group not in observed_coverage
            )
            if len(selected) + len(missing_coverage) > self.search_budget:
                raise RuntimeSearchError(
                    "rough placement coverage exceeds the search budget"
                )
            for group in missing_coverage:
                row = min(by_coverage[group], key=self._rank)
                if row.visit_id in selected_ids:
                    raise RuntimeSearchError(
                        "rough placement coverage selection is inconsistent"
                    )
                selected.append(replace(row, coverage_only=True))
                selected_ids.add(row.visit_id)
            selected.sort(key=lambda row: row.visit_id)
            frontier = RoughPlacementFrontier(
                artifact_sha256=artifact_sha256,
                capability_generation_sha256=capability_generation_sha256,
                input_token_bucket=input_bucket,
                output_token_bucket=output_bucket,
                search_budget=self.search_budget,
                visits=tuple(selected),
                rough_plan_count=len(rows),
            )
            if len(self._cache) >= self.maximum_entries:
                self._cache.pop(next(iter(self._cache)))
                self.evictions += 1
            self._cache[key] = frontier
            self.misses += 1
            self.generations += 1
            return frontier, False

    def stats(self) -> Mapping[str, int]:
        with self._lock:
            return MappingProxyType({
                "evictions": self.evictions,
                "generations": self.generations,
                "hits": self.hits,
                "misses": self.misses,
                "shape_bucket_entries": len(self._cache),
            })

    def invalidate_artifacts(self, artifact_sha256s: Sequence[str]) -> int:
        """Drop frontiers whose structural phone placement changed."""

        artifacts = frozenset(artifact_sha256s)
        if any(
            type(value) is not str
            or not value.startswith("sha256:")
            or len(value) != 71
            for value in artifacts
        ):
            raise RuntimeSearchError(
                "rough placement invalidation artifact is invalid"
            )
        with self._lock:
            keys = tuple(
                key for key in self._cache if key[0] in artifacts
            )
            for key in keys:
                del self._cache[key]
            return len(keys)
