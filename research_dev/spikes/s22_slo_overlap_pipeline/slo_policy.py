#!/usr/bin/env python3
"""Finite, profile-driven SLO route selection for the S22 proof runtime."""

from __future__ import annotations

import threading
from dataclasses import dataclass


MAX_US = (1 << 63) - 1


class NoFeasibleRoute(RuntimeError):
    pass


def _require_int(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > MAX_US:
        raise ValueError(f"{name} is out of range")


@dataclass(frozen=True)
class WorkRequest:
    request_id: int
    arrival_us: int
    slo_us: int
    steps: int
    priority: int

    def __post_init__(self) -> None:
        _require_int("request_id", self.request_id, 1)
        _require_int("arrival_us", self.arrival_us)
        _require_int("slo_us", self.slo_us, 1)
        _require_int("steps", self.steps, 1)
        _require_int("priority", self.priority)
        if self.arrival_us > MAX_US - self.slo_us:
            raise ValueError("request deadline overflows")

    @property
    def deadline_us(self) -> int:
        return self.arrival_us + self.slo_us


@dataclass(frozen=True)
class RouteProfile:
    route_id: str
    head_name: str
    offloaded_layers: int
    fixed_us: int
    p95_step_us: int
    profiled_batch: int
    max_active: int
    gather_cap_us: int
    wait_stages_per_step: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not self.route_id or not self.head_name:
            raise ValueError("route identifiers must be nonempty")
        _require_int("offloaded_layers", self.offloaded_layers)
        _require_int("fixed_us", self.fixed_us)
        _require_int("p95_step_us", self.p95_step_us, 1)
        _require_int("profiled_batch", self.profiled_batch, 1)
        _require_int("max_active", self.max_active, 1)
        _require_int("gather_cap_us", self.gather_cap_us)
        _require_int("wait_stages_per_step", self.wait_stages_per_step, 1)
        if self.profiled_batch > self.max_active:
            raise ValueError("profiled batch exceeds active capacity")
        if not self.evidence_sha256.startswith("sha256:") or len(self.evidence_sha256) != 71:
            raise ValueError("route profile requires a sha256 evidence digest")

    def service_us(self, steps: int) -> int:
        _require_int("steps", steps, 1)
        if steps > (MAX_US - self.fixed_us) // self.p95_step_us:
            raise ValueError("predicted service time overflows")
        return self.fixed_us + steps * self.p95_step_us


@dataclass(frozen=True)
class RouteDecision:
    request_id: int
    route_epoch: int
    route_id: str
    head_name: str
    predicted_finish_us: int
    deadline_us: int
    batch_wait_us: int
    reason: str


class SloRouter:
    def __init__(self, profiles: list[RouteProfile]):
        if not profiles:
            raise ValueError("at least one route profile is required")
        route_ids = [profile.route_id for profile in profiles]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route ids must be unique")
        self._profiles = {profile.route_id: profile for profile in profiles}
        self._active = {profile.route_id: 0 for profile in profiles}
        self._pinned: dict[int, RouteDecision] = {}
        self._next_epoch = 1
        self._lock = threading.Lock()

    def active_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._active)

    def pinned(self, request_id: int) -> RouteDecision | None:
        with self._lock:
            return self._pinned.get(request_id)

    def admit(self, request: WorkRequest, now_us: int) -> RouteDecision:
        _require_int("now_us", now_us)
        with self._lock:
            if request.request_id in self._pinned:
                raise ValueError("request is already pinned")
            start_us = max(now_us, request.arrival_us)
            candidates: list[tuple[RouteProfile, int, int]] = []
            for profile in self._profiles.values():
                active = self._active[profile.route_id]
                if active >= profile.max_active:
                    continue
                service_us = profile.service_us(request.steps)
                queued_waves = active // profile.profiled_batch
                queue_us = queued_waves * service_us
                if start_us > MAX_US - queue_us - service_us:
                    continue
                finish_us = start_us + queue_us + service_us
                if finish_us <= request.deadline_us:
                    slack_us = request.deadline_us - finish_us
                    candidates.append((profile, finish_us, slack_us))
            if not candidates:
                raise NoFeasibleRoute(f"no profiled route meets request {request.request_id} SLO")

            # Harvest the largest phone prefix first. Among equal-depth phone
            # routes, use the slowest feasible route and preserve faster
            # capacity for requests with less slack.
            profile, finish_us, slack_us = min(
                candidates,
                key=lambda item: (
                    -item[0].offloaded_layers,
                    -item[0].service_us(request.steps),
                    item[1],
                    item[0].route_id,
                ),
            )
            decision = RouteDecision(
                request.request_id,
                self._next_epoch,
                profile.route_id,
                profile.head_name,
                finish_us,
                request.deadline_us,
                min(
                    profile.gather_cap_us,
                    slack_us // (request.steps * profile.wait_stages_per_step),
                ),
                "MAX_OFFLOAD_SLOWEST_FEASIBLE",
            )
            self._next_epoch += 1
            self._active[profile.route_id] += 1
            self._pinned[request.request_id] = decision
            return decision

    def complete(self, request_id: int, route_epoch: int) -> RouteDecision:
        _require_int("request_id", request_id, 1)
        _require_int("route_epoch", route_epoch, 1)
        with self._lock:
            decision = self._pinned.get(request_id)
            if decision is None:
                raise ValueError("request is not pinned")
            if decision.route_epoch != route_epoch:
                raise ValueError("route epoch mismatch")
            if self._active[decision.route_id] <= 0:
                raise RuntimeError("route active count underflow")
            self._active[decision.route_id] -= 1
            del self._pinned[request_id]
            return decision
