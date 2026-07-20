#!/usr/bin/env python3
"""Fail-closed host coordinator for S14 priority-aware native batches.

This module owns queue, launch, and completion state. Device I/O is supplied by
the CP-D harness after a LAUNCH decision. It makes no latency or energy claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from power_frontier_policy import (
    BatchDecision,
    BatchPoint,
    BoundaryCertificate,
    CertifiedBatchPoint,
    PolicyError,
    WorkItem,
    choose_batch,
    throughput_knee,
)


class BatchRuntimeError(ValueError):
    pass


def _int(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise BatchRuntimeError(f"{name} must be an integer >= {minimum}")
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise BatchRuntimeError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class RouteConfig:
    route_id: str
    service_class: str
    model_id: str
    island_id: str
    profile_id: str
    route_epoch: int
    roofline_class: str
    points: tuple[BatchPoint, ...]
    high_priority_max: int = 0

    def validate(self) -> None:
        _text("route_id", self.route_id)
        _text("service_class", self.service_class)
        _text("model_id", self.model_id)
        _text("island_id", self.island_id)
        _text("profile_id", self.profile_id)
        _int("route_epoch", self.route_epoch, 1)
        _int("high_priority_max", self.high_priority_max)
        if self.roofline_class not in {"memory_bound", "compute_bound"}:
            raise BatchRuntimeError("roofline_class must be memory_bound or compute_bound")
        if type(self.points) is not tuple or not self.points:
            raise BatchRuntimeError("route profile is empty")
        if any(type(point) is not CertifiedBatchPoint for point in self.points):
            raise BatchRuntimeError("route profile contains an uncertified batch point")
        try:
            throughput_knee(self.points)
        except PolicyError as exc:
            raise BatchRuntimeError(str(exc)) from exc


@dataclass(frozen=True)
class Launch:
    launch_id: int
    route_id: str
    route_epoch: int
    profile_id: str
    compatibility_key: str
    request_ids: tuple[str, ...]
    batch_size: int
    predicted_duration_us: int
    start_us: int
    reason: str


@dataclass(frozen=True)
class Completion:
    request_id: str
    status: str
    finish_us: int
    launch_id: int


class PriorityBatchRuntime:
    """One bounded lane per route with exact launch/result ownership."""

    def __init__(self, routes: Sequence[RouteConfig]) -> None:
        if not routes:
            raise BatchRuntimeError("at least one route is required")
        self.routes: dict[str, RouteConfig] = {}
        for route in routes:
            if type(route) is not RouteConfig:
                raise BatchRuntimeError("routes must contain RouteConfig values")
            route.validate()
            if route.route_id in self.routes:
                raise BatchRuntimeError(f"duplicate route_id {route.route_id!r}")
            self.routes[route.route_id] = route
        self._pending: dict[str, tuple[str, WorkItem]] = {}
        self._inflight: dict[str, Launch] = {}
        self._launched_items: dict[tuple[int, str], WorkItem] = {}
        self._wait_until: dict[str, int] = {}
        self._completions: dict[str, Completion] = {}
        self._fallback: list[str] = []
        self._launch_counter = 0

    def enqueue(self, route_id: str, item: WorkItem, now_us: int) -> None:
        route = self._route(route_id)
        _int("now_us", now_us)
        if type(item) is not WorkItem:
            raise BatchRuntimeError("item must be a WorkItem")
        try:
            item.validate(now_us)
        except PolicyError as exc:
            raise BatchRuntimeError(str(exc)) from exc
        if (
            item.service_class != route.service_class
            or item.model_id != route.model_id
            or item.island_id != route.island_id
        ):
            raise BatchRuntimeError("work item does not match its route signature")
        if item.request_id in self._pending or item.request_id in self._completions:
            raise BatchRuntimeError(f"duplicate request_id {item.request_id!r}")
        if any(item.request_id in launch.request_ids for launch in self._inflight.values()):
            raise BatchRuntimeError(f"request_id {item.request_id!r} is already in flight")
        self._pending[item.request_id] = (route_id, item)

    def decide(self, route_id: str, now_us: int) -> BatchDecision | Launch:
        route = self._route(route_id)
        _int("now_us", now_us)
        if route_id in self._inflight:
            raise BatchRuntimeError(f"route {route_id!r} already has an in-flight launch")
        ready = [
            item
            for pending_route, item in self._pending.values()
            if pending_route == route_id
        ]
        try:
            decision = choose_batch(
                now_us,
                ready,
                route.points,
                route.roofline_class,
                route.high_priority_max,
            )
        except PolicyError as exc:
            raise BatchRuntimeError(str(exc)) from exc
        if decision.action != "LAUNCH":
            if decision.action == "WAIT":
                if decision.next_wake_us is None or decision.next_wake_us <= now_us:
                    raise BatchRuntimeError("WAIT decision has an invalid wake time")
                self._wait_until[route_id] = decision.next_wake_us
            else:
                self._wait_until.pop(route_id, None)
            return decision

        if decision.batch_size != len(decision.request_ids):
            raise BatchRuntimeError("policy launch has inconsistent batch ownership")
        if len(set(decision.request_ids)) != len(decision.request_ids):
            raise BatchRuntimeError("policy launch contains duplicate request ownership")
        if any(request_id not in self._pending for request_id in decision.request_ids):
            raise BatchRuntimeError("policy selected a request that is not pending")
        selected = [self._pending[request_id] for request_id in decision.request_ids]
        if any(pending_route != route_id for pending_route, _ in selected):
            raise BatchRuntimeError("policy crossed route ownership")
        self._launch_counter += 1
        launch = Launch(
            launch_id=self._launch_counter,
            route_id=route_id,
            route_epoch=route.route_epoch,
            profile_id=route.profile_id,
            compatibility_key=decision.compatibility_key or "",
            request_ids=decision.request_ids,
            batch_size=decision.batch_size,
            predicted_duration_us=decision.duration_us,
            start_us=now_us,
            reason=decision.reason,
        )
        for request_id, (_, item) in zip(launch.request_ids, selected):
            del self._pending[request_id]
            self._launched_items[(launch.launch_id, request_id)] = item
        self._inflight[route_id] = launch
        self._wait_until.pop(route_id, None)
        return launch

    def complete(
        self,
        route_id: str,
        launch_id: int,
        route_epoch: int,
        finish_us: int,
        certificates: Sequence[BoundaryCertificate],
    ) -> tuple[Completion, ...]:
        launch = self._owned_launch(route_id, launch_id, route_epoch)
        _int("finish_us", finish_us)
        if finish_us < launch.start_us:
            raise BatchRuntimeError("completion precedes launch")
        by_id: dict[str, BoundaryCertificate] = {}
        for cert in certificates:
            if type(cert) is not BoundaryCertificate:
                raise BatchRuntimeError("certificates must contain BoundaryCertificate values")
            _text("certificate.request_id", cert.request_id)
            if cert.request_id in by_id:
                raise BatchRuntimeError(f"duplicate certificate for {cert.request_id!r}")
            by_id[cert.request_id] = cert
        if set(by_id) != set(launch.request_ids):
            raise BatchRuntimeError("certificate set does not match launch ownership")

        admitted_by_id: dict[str, bool] = {}
        items_by_id = {
            request_id: self._item_from_launch(launch, request_id)
            for request_id in launch.request_ids
        }
        for request_id in launch.request_ids:
            try:
                admitted_by_id[request_id] = by_id[request_id].admitted()
            except PolicyError as exc:
                raise BatchRuntimeError(str(exc)) from exc

        out: list[Completion] = []
        for request_id in launch.request_ids:
            item = items_by_id[request_id]
            if not admitted_by_id[request_id]:
                status = "fallback_required"
                self._fallback.append(request_id)
            elif finish_us > item.deadline_us:
                status = "tardy_result"
            else:
                status = "completed"
            completion = Completion(request_id, status, finish_us, launch_id)
            self._completions[request_id] = completion
            del self._launched_items[(launch.launch_id, request_id)]
            out.append(completion)
        del self._inflight[route_id]
        return tuple(out)

    def fail_launch(
        self,
        route_id: str,
        launch_id: int,
        route_epoch: int,
        finish_us: int,
    ) -> tuple[Completion, ...]:
        launch = self._owned_launch(route_id, launch_id, route_epoch)
        _int("finish_us", finish_us)
        if finish_us < launch.start_us:
            raise BatchRuntimeError("failure precedes launch")
        for request_id in launch.request_ids:
            self._item_from_launch(launch, request_id)
        out = []
        for request_id in launch.request_ids:
            completion = Completion(request_id, "fallback_required", finish_us, launch_id)
            self._completions[request_id] = completion
            self._fallback.append(request_id)
            del self._launched_items[(launch.launch_id, request_id)]
            out.append(completion)
        del self._inflight[route_id]
        return tuple(out)

    def next_wake_us(self) -> int | None:
        return min(self._wait_until.values()) if self._wait_until else None

    def fallback_request_ids(self) -> tuple[str, ...]:
        return tuple(self._fallback)

    def snapshot(self) -> dict[str, object]:
        return {
            "pending": tuple(sorted(self._pending)),
            "inflight": tuple(
                (route_id, launch.launch_id, launch.request_ids)
                for route_id, launch in sorted(self._inflight.items())
            ),
            "completed": tuple(
                (request_id, completion.status)
                for request_id, completion in sorted(self._completions.items())
            ),
            "fallback": tuple(self._fallback),
            "next_wake_us": self.next_wake_us(),
        }

    def _route(self, route_id: str) -> RouteConfig:
        _text("route_id", route_id)
        try:
            return self.routes[route_id]
        except KeyError as exc:
            raise BatchRuntimeError(f"unknown route_id {route_id!r}") from exc

    def _owned_launch(self, route_id: str, launch_id: int, route_epoch: int) -> Launch:
        route = self._route(route_id)
        _int("launch_id", launch_id, 1)
        _int("route_epoch", route_epoch, 1)
        launch = self._inflight.get(route_id)
        if launch is None:
            raise BatchRuntimeError(f"route {route_id!r} has no in-flight launch")
        if launch.launch_id != launch_id:
            raise BatchRuntimeError("stale or foreign launch_id")
        if route.route_epoch != route_epoch or launch.route_epoch != route_epoch:
            raise BatchRuntimeError("stale route_epoch")
        return launch

    def _item_from_launch(self, launch: Launch, request_id: str) -> WorkItem:
        try:
            return self._launched_items[(launch.launch_id, request_id)]
        except KeyError as exc:
            raise BatchRuntimeError("launched item ledger is inconsistent") from exc
