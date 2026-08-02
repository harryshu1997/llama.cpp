#!/usr/bin/env python3
"""Bounded priority, latest-start, relief, and resource admission policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


MAX_US = (1 << 63) - 1


class PriorityPolicyError(RuntimeError):
    pass


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > MAX_US:
        raise PriorityPolicyError(f"{name} must be an integer in [{minimum},{MAX_US}]")
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise PriorityPolicyError(f"{name} must be a nonempty string")
    return value


def _digest(name: str, value: object) -> str:
    text = _text(name, value)
    if (
        not text.startswith("sha256:")
        or len(text) != 71
        or any(char not in "0123456789abcdef" for char in text[7:])
    ):
        raise PriorityPolicyError(f"{name} must be sha256:<64 lowercase hex>")
    return text


@dataclass(frozen=True)
class PriorityWork:
    request_id: int
    arrival_us: int
    deadline_us: int
    priority: int
    input_tokens: int
    output_steps: int

    def validate(self, now_us: int) -> None:
        _integer("request_id", self.request_id, 1)
        _integer("arrival_us", self.arrival_us)
        _integer("deadline_us", self.deadline_us, 1)
        _integer("priority", self.priority)
        _integer("input_tokens", self.input_tokens, 1)
        _integer("output_steps", self.output_steps, 1)
        _integer("now_us", now_us)
        if self.deadline_us <= self.arrival_us:
            raise PriorityPolicyError("deadline must follow arrival")
        if self.arrival_us > now_us:
            raise PriorityPolicyError("future request cannot enter READY")


@dataclass(frozen=True)
class RouteBatchPoint:
    batch_size: int
    input_tokens: int
    output_steps: int
    duration_us: int
    cuda_work_us: int
    evidence_sha256: tuple[str, ...]
    derivation: str

    def validate(self) -> None:
        _integer("batch_size", self.batch_size, 1)
        _integer("input_tokens", self.input_tokens, 1)
        _integer("output_steps", self.output_steps, 1)
        _integer("duration_us", self.duration_us, 1)
        _integer("cuda_work_us", self.cuda_work_us, 1)
        if (
            type(self.evidence_sha256) is not tuple
            or not self.evidence_sha256
            or len(set(self.evidence_sha256)) != len(self.evidence_sha256)
        ):
            raise PriorityPolicyError("evidence digests must be unique and nonempty")
        for digest in self.evidence_sha256:
            _digest("evidence_sha256", digest)
        _text("derivation", self.derivation)


@dataclass(frozen=True)
class PriorityRoute:
    route_id: str
    resources: tuple[str, ...]
    points: tuple[RouteBatchPoint, ...]
    gather_us: int

    def validate(self) -> None:
        _text("route_id", self.route_id)
        _integer("gather_us", self.gather_us)
        if not self.resources or len(set(self.resources)) != len(self.resources):
            raise PriorityPolicyError("route resources must be unique and nonempty")
        for resource in self.resources:
            _text("resource", resource)
        if not self.points:
            raise PriorityPolicyError("route points must be nonempty")
        seen = set()
        for point in self.points:
            if type(point) is not RouteBatchPoint:
                raise PriorityPolicyError("route point type is invalid")
            point.validate()
            if point.batch_size in seen:
                raise PriorityPolicyError("route batch sizes must be unique")
            seen.add(point.batch_size)
        if 1 not in seen:
            raise PriorityPolicyError("route must include batch size 1")

    def point(self, batch_size: int) -> RouteBatchPoint:
        for point in self.points:
            if point.batch_size == batch_size:
                return point
        raise PriorityPolicyError(
            f"route {self.route_id} has no measured B{batch_size} point"
        )

    def largest_at_most(self, count: int) -> RouteBatchPoint:
        available = [point for point in self.points if point.batch_size <= count]
        if not available:
            raise PriorityPolicyError("no measured point fits the ready count")
        return max(available, key=lambda point: point.batch_size)

    def target(self) -> RouteBatchPoint:
        return max(self.points, key=lambda point: point.batch_size)


@dataclass(frozen=True)
class DispatchGroup:
    route_id: str
    request_ids: tuple[int, ...]
    route_epochs: tuple[int, ...]
    batch_size: int
    predicted_start_us: int
    predicted_finish_us: int
    batch_wait_us: int
    predicted_cuda_relief_us: int
    reason: str


@dataclass(frozen=True)
class WaitDecision:
    next_wake_us: int | None
    reason: str


@dataclass(frozen=True)
class RejectDecision:
    request_id: int
    reason: str


@dataclass(frozen=True)
class Reservation:
    request_id: int
    route_epoch: int
    route_id: str
    priority: int


class PriorityAdmissionController:
    def __init__(
        self,
        routes: Sequence[PriorityRoute],
        capacities: Mapping[str, int],
        urgent_reserve: Mapping[str, int],
        urgent_priority_max: int = 0,
        fallback_route_id: str = "R0",
        offload_route_id: str = "R2",
        offload_enabled: bool = True,
    ):
        _integer("urgent_priority_max", urgent_priority_max)
        if type(offload_enabled) is not bool:
            raise PriorityPolicyError("offload_enabled must be boolean")
        if not routes:
            raise PriorityPolicyError("route set must be nonempty")
        self.routes = {}
        for route in routes:
            if type(route) is not PriorityRoute:
                raise PriorityPolicyError("route type is invalid")
            route.validate()
            if route.route_id in self.routes:
                raise PriorityPolicyError("route id is duplicated")
            self.routes[route.route_id] = route
        if set(self.routes) != {fallback_route_id, offload_route_id}:
            raise PriorityPolicyError("policy requires exactly fallback and offload routes")
        self.fallback_route_id = fallback_route_id
        self.offload_route_id = offload_route_id
        self.urgent_priority_max = urgent_priority_max
        self.offload_enabled = offload_enabled

        required = {resource for route in routes for resource in route.resources}
        if set(capacities) != required or set(urgent_reserve) != required:
            raise PriorityPolicyError("capacity maps must exactly cover route resources")
        self.capacities = {}
        self.urgent_reserve = {}
        for resource in sorted(required):
            capacity = _integer(f"capacity.{resource}", capacities[resource], 1)
            reserve = _integer(f"urgent_reserve.{resource}", urgent_reserve[resource])
            if reserve > capacity:
                raise PriorityPolicyError("urgent reserve exceeds capacity")
            self.capacities[resource] = capacity
            self.urgent_reserve[resource] = reserve

        self._pending: dict[int, PriorityWork] = {}
        self._active: dict[int, Reservation] = {}
        self._resource_active = {resource: 0 for resource in required}
        self._resource_low = {resource: 0 for resource in required}
        self._next_epoch = 1

    def enqueue(self, work: PriorityWork, now_us: int) -> None:
        if type(work) is not PriorityWork:
            raise PriorityPolicyError("work type is invalid")
        work.validate(now_us)
        if work.request_id in self._pending or work.request_id in self._active:
            raise PriorityPolicyError("request is already owned")
        self._pending[work.request_id] = work

    def pending(self) -> tuple[PriorityWork, ...]:
        return tuple(sorted(self._pending.values(), key=self._order_key))

    def active(self) -> dict[int, Reservation]:
        return dict(self._active)

    def resource_state(self) -> dict[str, dict[str, int]]:
        return {
            "capacity": dict(self.capacities),
            "urgent_reserve": dict(self.urgent_reserve),
            "active": dict(self._resource_active),
            "low_priority_active": dict(self._resource_low),
        }

    @staticmethod
    def _order_key(work: PriorityWork) -> tuple[int, int, int, int]:
        return work.priority, work.deadline_us, work.arrival_us, work.request_id

    def _has_urgent(self) -> bool:
        return any(
            work.priority <= self.urgent_priority_max
            for work in self._pending.values()
        ) or any(
            reservation.priority <= self.urgent_priority_max
            for reservation in self._active.values()
        )

    def _can_reserve(
        self, route: PriorityRoute, count: int, priority: int, borrow: bool = False,
    ) -> bool:
        urgent = priority <= self.urgent_priority_max
        for resource in route.resources:
            if self._resource_active[resource] + count > self.capacities[resource]:
                return False
            if not urgent and not borrow and (
                self._resource_low[resource] + count
                > self.capacities[resource] - self.urgent_reserve[resource]
            ):
                return False
        return True

    @staticmethod
    def _fits_deadlines(
        works: Sequence[PriorityWork], now_us: int, point: RouteBatchPoint,
    ) -> bool:
        if any(
            work.input_tokens != point.input_tokens
            or work.output_steps != point.output_steps
            for work in works
        ):
            return False
        if now_us > MAX_US - point.duration_us:
            return False
        finish = now_us + point.duration_us
        return all(finish <= work.deadline_us for work in works)

    def _relief(self, point: RouteBatchPoint) -> int:
        fallback_route = self.routes[self.fallback_route_id]
        try:
            fallback = fallback_route.point(point.batch_size)
            baseline = fallback.cuda_work_us
        except PriorityPolicyError:
            fallback = fallback_route.point(1)
            if fallback.cuda_work_us > MAX_US // point.batch_size:
                raise PriorityPolicyError("CUDA baseline overflows the scheduler domain")
            baseline = fallback.cuda_work_us * point.batch_size
        return baseline - point.cuda_work_us

    def _reserve(
        self,
        route: PriorityRoute,
        works: Sequence[PriorityWork],
        point: RouteBatchPoint,
        now_us: int,
        relief_us: int,
        reason: str,
        borrow: bool = False,
        batch_wait_us: int | None = None,
    ) -> DispatchGroup:
        if len(works) != point.batch_size:
            raise PriorityPolicyError("dispatch ownership differs from batch point")
        priority = min(work.priority for work in works)
        if not self._can_reserve(route, len(works), priority, borrow):
            raise PriorityPolicyError("dispatch attempted without resource capacity")
        if not self._fits_deadlines(works, now_us, point):
            raise PriorityPolicyError("dispatch attempted past a deadline")
        epochs = []
        for work in works:
            if self._next_epoch > MAX_US:
                raise PriorityPolicyError("route epoch exhausted")
            epoch = self._next_epoch
            self._next_epoch += 1
            epochs.append(epoch)
            self._active[work.request_id] = Reservation(
                work.request_id, epoch, route.route_id, work.priority,
            )
            del self._pending[work.request_id]
            for resource in route.resources:
                self._resource_active[resource] += 1
                if work.priority > self.urgent_priority_max:
                    self._resource_low[resource] += 1
        return DispatchGroup(
            route.route_id,
            tuple(work.request_id for work in works),
            tuple(epochs),
            point.batch_size,
            now_us,
            now_us + point.duration_us,
            route.gather_us if batch_wait_us is None else _integer(
                "batch_wait_us", batch_wait_us,
            ),
            relief_us,
            reason,
        )

    def _fallback(
        self, work: PriorityWork, now_us: int, reason: str,
    ) -> DispatchGroup | WaitDecision | RejectDecision:
        route = self.routes[self.fallback_route_id]
        point = route.point(1)
        if not self._fits_deadlines((work,), now_us, point):
            del self._pending[work.request_id]
            return RejectDecision(work.request_id, "NO_MEASURED_ROUTE_CAN_MEET_SLO")
        urgent = work.priority <= self.urgent_priority_max
        borrow = not urgent and not self._has_urgent()
        if not self._can_reserve(route, 1, work.priority, borrow):
            latest = work.deadline_us - point.duration_us
            return WaitDecision(max(now_us, latest), "FALLBACK_RESOURCE_RESERVED")
        return self._reserve(
            route, (work,), point, now_us, 0, reason, borrow, 0,
        )

    def decide(
        self, now_us: int,
    ) -> DispatchGroup | WaitDecision | RejectDecision:
        _integer("now_us", now_us)
        ordered = list(self.pending())
        if not ordered:
            return WaitDecision(None, "READY_QUEUE_EMPTY")

        urgent = [
            work for work in ordered
            if work.priority <= self.urgent_priority_max
        ]
        if urgent:
            route = self.routes[self.fallback_route_id]
            point = route.largest_at_most(len(urgent))
            selected = urgent[:point.batch_size]
            if not self._fits_deadlines(selected, now_us, point):
                return self._fallback(urgent[0], now_us, "URGENT_R0_FALLBACK")
            if not self._can_reserve(route, len(selected), urgent[0].priority):
                latest = min(work.deadline_us for work in selected) - point.duration_us
                return WaitDecision(max(now_us, latest), "URGENT_RESOURCE_BUSY")
            return self._reserve(
                route, selected, point, now_us, 0, "URGENT_PRIORITY_R0",
                batch_wait_us=0 if point.batch_size == 1 else route.gather_us,
            )

        if not self.offload_enabled:
            route = self.routes[self.fallback_route_id]
            point = route.largest_at_most(len(ordered))
            selected = ordered[:point.batch_size]
            if not self._fits_deadlines(selected, now_us, point):
                return self._fallback(
                    selected[0], now_us, "CONTROL_LATEST_START_FALLBACK",
                )
            borrow = not self._has_urgent()
            if not self._can_reserve(
                route, len(selected), selected[0].priority, borrow,
            ):
                latest = min(work.deadline_us for work in selected) - point.duration_us
                return WaitDecision(max(now_us, latest), "CONTROL_CUDA_RESOURCE_BUSY")
            return self._reserve(
                route, selected, point, now_us, 0,
                "ALL_CUDA_MATCHED_CONTROL", borrow,
                0 if point.batch_size == 1 else route.gather_us,
            )

        route = self.routes[self.offload_route_id]
        target = route.target()
        selected = ordered[:target.batch_size]
        target_latest = min(work.deadline_us for work in selected) - target.duration_us
        relief = self._relief(target)
        if len(selected) == target.batch_size:
            if relief <= 0:
                return self._fallback(
                    selected[0], now_us, "OFFLOAD_HAS_NO_SERVER_RELIEF",
                )
            if not self._fits_deadlines(selected, now_us, target):
                return self._fallback(
                    selected[0], now_us, "OFFLOAD_CANNOT_MEET_SLO",
                )
            if not self._can_reserve(route, len(selected), selected[0].priority):
                return WaitDecision(
                    max(now_us, target_latest), "OFFLOAD_RESOURCE_BUSY",
                )
            return self._reserve(
                route, selected, target, now_us, relief,
                "SERVER_RELIEVING_TARGET_BATCH",
            )

        if now_us < target_latest:
            return WaitDecision(target_latest, "BOUNDED_WAIT_FOR_TARGET_BATCH")
        current = route.largest_at_most(len(selected))
        current_works = selected[:current.batch_size]
        current_relief = self._relief(current)
        if current_relief <= 0 or not self._fits_deadlines(
            current_works, now_us, current,
        ):
            return self._fallback(
                selected[0], now_us, "LATEST_START_SERVER_FALLBACK",
            )
        if not self._can_reserve(
            route, len(current_works), current_works[0].priority,
        ):
            return self._fallback(
                selected[0], now_us, "LATEST_START_RESOURCE_FALLBACK",
            )
        return self._reserve(
            route, current_works, current, now_us, current_relief,
            "LATEST_START_MEASURED_SMALL_BATCH",
        )

    def complete(self, request_id: int, route_epoch: int) -> Reservation:
        return self.complete_group((request_id,), (route_epoch,))[0]

    def complete_group(
        self, request_ids: Sequence[int], route_epochs: Sequence[int],
    ) -> tuple[Reservation, ...]:
        if (
            not request_ids
            or len(request_ids) != len(route_epochs)
            or len(set(request_ids)) != len(request_ids)
        ):
            raise PriorityPolicyError("completion group identity is invalid")
        reservations = []
        for request_id, route_epoch in zip(request_ids, route_epochs):
            _integer("request_id", request_id, 1)
            _integer("route_epoch", route_epoch, 1)
            reservation = self._active.get(request_id)
            if reservation is None:
                raise PriorityPolicyError("request is not active")
            if reservation.route_epoch != route_epoch:
                raise PriorityPolicyError("route epoch mismatch")
            reservations.append(reservation)
        decrements = {resource: 0 for resource in self._resource_active}
        low_decrements = {resource: 0 for resource in self._resource_low}
        for reservation in reservations:
            route = self.routes[reservation.route_id]
            for resource in route.resources:
                decrements[resource] += 1
                if reservation.priority > self.urgent_priority_max:
                    low_decrements[resource] += 1
        if any(
            self._resource_active[resource] < count
            for resource, count in decrements.items()
        ) or any(
            self._resource_low[resource] < count
            for resource, count in low_decrements.items()
        ):
            raise PriorityPolicyError("resource count underflow")
        for resource, count in decrements.items():
            self._resource_active[resource] -= count
            self._resource_low[resource] -= low_decrements[resource]
        for reservation in reservations:
            del self._active[reservation.request_id]
        return tuple(reservations)
