#!/usr/bin/env python3
"""Deterministic SLO-aware route selection over the finite S36 routes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence


SERVER_ROUTE_ID = "cuda-c4"


class PolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class RouteProfile:
    route_id: str
    device: str
    cut: int
    predicted_p95_us: int
    safety_margin_us: int
    measured: bool
    eligible: bool

    def validate(self) -> None:
        if (
            not self.route_id
            or not self.device
            or type(self.cut) is not int
            or self.cut <= 0
            or type(self.predicted_p95_us) is not int
            or self.predicted_p95_us <= 0
            or type(self.safety_margin_us) is not int
            or self.safety_margin_us < 0
            or type(self.measured) is not bool
            or type(self.eligible) is not bool
        ):
            raise PolicyError("invalid route profile")
        if self.eligible and not self.measured:
            raise PolicyError("an unmeasured route cannot be eligible")


@dataclass(frozen=True)
class DeviceState:
    active_requests: int
    capacity: int
    estimated_queue_us: int
    ready: bool
    queued_rows_by_cut: Mapping[int, int]

    def validate(self) -> None:
        if (
            type(self.active_requests) is not int
            or type(self.capacity) is not int
            or type(self.estimated_queue_us) is not int
            or self.active_requests < 0
            or self.capacity <= 0
            or self.active_requests > self.capacity
            or self.estimated_queue_us < 0
            or type(self.ready) is not bool
        ):
            raise PolicyError("invalid device state")
        for cut, rows in self.queued_rows_by_cut.items():
            if type(cut) is not int or cut <= 0 or type(rows) is not int or rows < 0:
                raise PolicyError("invalid queued-row state")


@dataclass(frozen=True)
class RequestWork:
    request_id: int
    priority: int
    arrival_us: int
    deadline_us: int

    def validate(self) -> None:
        if (
            type(self.request_id) is not int
            or self.request_id < 0
            or type(self.priority) is not int
            or self.priority < 0
            or type(self.arrival_us) is not int
            or self.arrival_us < 0
            or type(self.deadline_us) is not int
            or self.deadline_us <= self.arrival_us
        ):
            raise PolicyError("invalid request work")


@dataclass(frozen=True)
class RouteDecision:
    request_id: int
    route_id: str
    device: str
    cut: int
    predicted_finish_us: int
    reason: str


class DynamicCutPolicy:
    def __init__(self, profiles: Sequence[RouteProfile]) -> None:
        by_id: dict[str, RouteProfile] = {}
        for profile in profiles:
            profile.validate()
            if profile.route_id in by_id:
                raise PolicyError("route profile is duplicated")
            by_id[profile.route_id] = profile
        if SERVER_ROUTE_ID not in by_id:
            raise PolicyError("server fallback profile is missing")
        server = by_id[SERVER_ROUTE_ID]
        if server.device != "cuda" or not server.measured or not server.eligible:
            raise PolicyError("server fallback must be measured and eligible")
        self._profiles = by_id

    @property
    def profiles(self) -> Mapping[str, RouteProfile]:
        return dict(self._profiles)

    def choose(
        self,
        work: RequestWork,
        now_us: int,
        devices: Mapping[str, DeviceState],
        force_control: bool = False,
    ) -> RouteDecision:
        work.validate()
        if type(now_us) is not int or now_us < work.arrival_us:
            raise PolicyError("invalid scheduling time")
        for state in devices.values():
            state.validate()

        if force_control:
            return self._server(work, now_us, "CONTROL_ALL_CUDA")
        if work.priority == 0:
            return self._server(work, now_us, "PRIORITY_ZERO_CUDA")

        remaining_us = work.deadline_us - now_us
        candidates: list[tuple[tuple[object, ...], RouteProfile, int]] = []
        for profile in self._profiles.values():
            if profile.device == "cuda" or not profile.measured or not profile.eligible:
                continue
            state = devices.get(profile.device)
            if state is None or not state.ready or state.active_requests >= state.capacity:
                continue
            predicted_us = (
                profile.predicted_p95_us
                + profile.safety_margin_us
                + state.estimated_queue_us
            )
            if predicted_us > remaining_us:
                continue
            queued_same_cut = state.queued_rows_by_cut.get(profile.cut, 0)
            normalized_load = Fraction(state.active_requests, state.capacity)
            if work.priority == 1:
                key: tuple[object, ...] = (
                    predicted_us,
                    -queued_same_cut,
                    normalized_load,
                    -profile.cut,
                    profile.route_id,
                )
            else:
                key = (
                    -profile.cut,
                    -queued_same_cut,
                    normalized_load,
                    predicted_us,
                    profile.route_id,
                )
            candidates.append((key, profile, predicted_us))
        if not candidates:
            return self._server(work, now_us, "NO_FEASIBLE_PHONE_ROUTE")
        candidates.sort(key=lambda item: item[0])
        profile = candidates[0][1]
        predicted_us = candidates[0][2]
        return RouteDecision(
            request_id=work.request_id,
            route_id=profile.route_id,
            device=profile.device,
            cut=profile.cut,
            predicted_finish_us=now_us + predicted_us,
            reason=(
                "FASTEST_FEASIBLE_PHONE_ROUTE"
                if work.priority == 1
                else "DEEPEST_FEASIBLE_PHONE_ROUTE"
            ),
        )

    def _server(
        self, work: RequestWork, now_us: int, reason: str,
    ) -> RouteDecision:
        profile = self._profiles[SERVER_ROUTE_ID]
        predicted_us = profile.predicted_p95_us + profile.safety_margin_us
        return RouteDecision(
            request_id=work.request_id,
            route_id=profile.route_id,
            device=profile.device,
            cut=profile.cut,
            predicted_finish_us=now_us + predicted_us,
            reason=reason,
        )
