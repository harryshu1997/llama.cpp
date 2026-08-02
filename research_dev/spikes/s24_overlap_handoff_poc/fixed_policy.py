#!/usr/bin/env python3
"""Finite stateful SLO policy for the S24 R0/R1/R2 routes."""

from __future__ import annotations

import threading
from dataclasses import dataclass


MAX_US = (1 << 63) - 1
FIXED_ROUTE_IDS = ("R0", "R1", "R2")


class NoFeasibleRoute(RuntimeError):
    pass


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > MAX_US:
        raise ValueError(f"{name} is out of range")


@dataclass(frozen=True)
class SloWork:
    request_id: int
    arrival_us: int
    slo_us: int
    prompt_tokens: int
    output_steps: int
    priority: int

    def __post_init__(self) -> None:
        _integer("request_id", self.request_id, 1)
        _integer("arrival_us", self.arrival_us)
        _integer("slo_us", self.slo_us, 1)
        _integer("prompt_tokens", self.prompt_tokens, 1)
        _integer("output_steps", self.output_steps, 1)
        _integer("priority", self.priority)
        if self.arrival_us > MAX_US - self.slo_us:
            raise ValueError("request deadline overflows")

    @property
    def deadline_us(self) -> int:
        return self.arrival_us + self.slo_us


@dataclass(frozen=True)
class FixedRouteProfile:
    route_id: str
    offload_rank: int
    fixed_us: int
    prefill_token_us: int
    decode_step_us: int
    profiled_batch: int
    max_active: int
    gather_cap_us: int
    resources: tuple[str, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.route_id not in FIXED_ROUTE_IDS:
            raise ValueError("profile route is outside the fixed route set")
        _integer("offload_rank", self.offload_rank)
        _integer("fixed_us", self.fixed_us)
        _integer("prefill_token_us", self.prefill_token_us, 1)
        _integer("decode_step_us", self.decode_step_us, 1)
        _integer("profiled_batch", self.profiled_batch, 1)
        _integer("max_active", self.max_active, 1)
        _integer("gather_cap_us", self.gather_cap_us)
        if self.profiled_batch > self.max_active:
            raise ValueError("profiled batch exceeds route active capacity")
        if not self.resources or len(self.resources) != len(set(self.resources)):
            raise ValueError("route resources must be unique and nonempty")
        if any(not resource for resource in self.resources):
            raise ValueError("route resource name must be nonempty")
        if (
            not self.evidence_sha256.startswith("sha256:")
            or len(self.evidence_sha256) != 71
        ):
            raise ValueError("route profile requires a SHA256 evidence digest")

    def service_us(self, work: SloWork) -> int:
        decode_rows = max(0, work.output_steps - 1)
        terms = (
            self.fixed_us,
            work.prompt_tokens * self.prefill_token_us,
            decode_rows * self.decode_step_us,
        )
        result = sum(terms)
        if result > MAX_US:
            raise ValueError("predicted service time overflows")
        return result

    def wait_points(self, work: SloWork) -> int:
        rows = work.prompt_tokens + max(0, work.output_steps - 1)
        return rows * len(self.resources)


@dataclass(frozen=True)
class FixedRouteDecision:
    request_id: int
    route_epoch: int
    route_id: str
    predicted_finish_us: int
    deadline_us: int
    batch_wait_us: int
    reason: str


class FixedSloRouter:
    def __init__(
        self,
        profiles: list[FixedRouteProfile],
        resource_capacities: dict[str, int],
    ):
        if {profile.route_id for profile in profiles} != set(FIXED_ROUTE_IDS):
            raise ValueError("profiles must define exactly R0, R1, and R2")
        if len(profiles) != len(FIXED_ROUTE_IDS):
            raise ValueError("route profiles must be unique")
        if not resource_capacities:
            raise ValueError("resource capacities must be nonempty")
        for resource, capacity in resource_capacities.items():
            if not resource:
                raise ValueError("resource name must be nonempty")
            _integer(f"capacity for {resource}", capacity, 1)
        required = {
            resource for profile in profiles for resource in profile.resources
        }
        if not required.issubset(resource_capacities):
            raise ValueError("a route resource has no capacity")
        self._profiles = {profile.route_id: profile for profile in profiles}
        self._route_active = {
            route_id: 0 for route_id in FIXED_ROUTE_IDS
        }
        self._resource_active = {
            resource: 0 for resource in resource_capacities
        }
        self._resource_capacities = dict(resource_capacities)
        self._pinned: dict[int, FixedRouteDecision] = {}
        self._next_epoch = 1
        self._lock = threading.Lock()

    def pinned(self, request_id: int) -> FixedRouteDecision | None:
        with self._lock:
            return self._pinned.get(request_id)

    def active_counts(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {
                "routes": dict(self._route_active),
                "resources": dict(self._resource_active),
            }

    def admit(self, work: SloWork, now_us: int) -> FixedRouteDecision:
        _integer("now_us", now_us)
        with self._lock:
            if work.request_id in self._pinned:
                raise ValueError("request is already pinned")
            start_us = max(now_us, work.arrival_us)
            candidates: list[
                tuple[FixedRouteProfile, int, int, int]
            ] = []
            for profile in self._profiles.values():
                if self._route_active[profile.route_id] >= profile.max_active:
                    continue
                if any(
                    self._resource_active[resource]
                    >= self._resource_capacities[resource]
                    for resource in profile.resources
                ):
                    continue
                service_us = profile.service_us(work)
                route_waves = (
                    self._route_active[profile.route_id]
                    // profile.profiled_batch
                )
                resource_waves = max(
                    self._resource_active[resource]
                    // profile.profiled_batch
                    for resource in profile.resources
                )
                queued_waves = max(route_waves, resource_waves)
                queue_us = queued_waves * service_us
                if start_us > MAX_US - queue_us - service_us:
                    continue
                finish_us = start_us + queue_us + service_us
                if finish_us <= work.deadline_us:
                    candidates.append((
                        profile,
                        finish_us,
                        work.deadline_us - finish_us,
                        service_us,
                    ))
            if not candidates:
                raise NoFeasibleRoute(
                    f"no fixed route meets request {work.request_id} SLO"
                )

            profile, finish_us, slack_us, _service_us = min(
                candidates,
                key=lambda item: (
                    -item[0].offload_rank,
                    -item[3],
                    item[1],
                    item[0].route_id,
                ),
            )
            decision = FixedRouteDecision(
                request_id=work.request_id,
                route_epoch=self._next_epoch,
                route_id=profile.route_id,
                predicted_finish_us=finish_us,
                deadline_us=work.deadline_us,
                batch_wait_us=min(
                    profile.gather_cap_us,
                    slack_us // profile.wait_points(work),
                ),
                reason="MAX_OFFLOAD_SLOWEST_FEASIBLE_FIXED_ROUTE",
            )
            self._next_epoch += 1
            self._route_active[profile.route_id] += 1
            for resource in profile.resources:
                self._resource_active[resource] += 1
            self._pinned[work.request_id] = decision
            return decision

    def complete(
        self, request_id: int, route_epoch: int,
    ) -> FixedRouteDecision:
        _integer("request_id", request_id, 1)
        _integer("route_epoch", route_epoch, 1)
        with self._lock:
            decision = self._pinned.get(request_id)
            if decision is None:
                raise ValueError("request is not pinned")
            if decision.route_epoch != route_epoch:
                raise ValueError("route epoch mismatch")
            profile = self._profiles[decision.route_id]
            if self._route_active[decision.route_id] <= 0:
                raise RuntimeError("route active count underflow")
            self._route_active[decision.route_id] -= 1
            for resource in profile.resources:
                if self._resource_active[resource] <= 0:
                    raise RuntimeError("resource active count underflow")
                self._resource_active[resource] -= 1
            del self._pinned[request_id]
            return decision
