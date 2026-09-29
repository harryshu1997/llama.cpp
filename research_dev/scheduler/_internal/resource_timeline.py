"""Resource lease calendar: lease records, previews, checkpoints and the ResourceTimeline owner."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Mapping, Sequence
from .policy_common import (
    SchedulerError as SchedulerError,
    _strict_bool as _strict_bool,
    _strict_int as _strict_int,
    _text as _text,
    ResourceProfile as ResourceProfile,
    LeaseDemand as LeaseDemand,
)


@dataclass(frozen=True)
class LeasePlan:
    lease_id: str
    resource_id: str
    lanes: tuple[int, ...]
    start_us: int
    predicted_end_us: int
    reserved_until_us: int


@dataclass(frozen=True)
class LeasePreview:
    start_us: int
    finish_us: int
    finish_upper_us: int
    plans: tuple[LeasePlan, ...]
    queue_by_resource_us: Mapping[str, int]
    blocking_resources: tuple[str, ...]


@dataclass(frozen=True)
class LeaseRecord:
    token: str
    owner_id: str
    lease_id: str
    resource_id: str
    lanes: tuple[int, ...]
    start_us: int
    predicted_end_us: int
    reserved_until_us: int


@dataclass
class _CalendarReservation:
    token: str
    owner_id: str
    lease_id: str
    start_us: int
    end_us: int


@dataclass(frozen=True)
class _ResourceTimelineCheckpoint:
    timeline_identity: int
    resources: tuple[tuple[str, ResourceProfile], ...]
    ready: tuple[tuple[str, bool], ...]
    calendar: tuple[
        tuple[str, tuple[tuple[_CalendarReservation, ...], ...]], ...
    ]
    tokens: tuple[
        tuple[str, tuple[_CalendarReservation, ...]], ...
    ]
    reservation_ends: tuple[tuple[_CalendarReservation, int], ...]
    next_token: int


class ResourceTimeline:
    def __init__(
        self,
        resources: Mapping[str, ResourceProfile],
        ready_overrides: Mapping[str, bool] | None = None,
    ) -> None:
        self._resources = dict(resources)
        self._ready = {
            resource_id: resource.ready
            for resource_id, resource in resources.items()
        }
        if ready_overrides is not None:
            for resource_id, ready in ready_overrides.items():
                if resource_id not in resources:
                    raise SchedulerError("readiness override references an unknown resource")
                self._ready[resource_id] = _strict_bool(
                    f"readiness override {resource_id}", ready
                )
        self._calendar: dict[str, list[list[_CalendarReservation]]] = {
            resource_id: [[] for _ in range(resource.capacity)]
            for resource_id, resource in resources.items()
        }
        self._tokens: dict[str, list[_CalendarReservation]] = {}
        self._next_token = 1

    def register_resources(
        self, resources: Mapping[str, ResourceProfile]
    ) -> None:
        """Add discovered resources without replacing an existing timeline."""
        for resource_id, resource in sorted(resources.items()):
            if not isinstance(resource, ResourceProfile):
                raise SchedulerError("registered resource profile is invalid")
            current = self._resources.get(resource_id)
            if current is not None:
                if current != resource:
                    raise SchedulerError(
                        f"registered resource differs: {resource_id}"
                    )
                continue
            self._resources[resource_id] = resource
            self._ready[resource_id] = resource.ready
            self._calendar[resource_id] = [
                [] for _ in range(resource.capacity)
            ]

    def checkpoint(self) -> object:
        reservations = {
            id(reservation): reservation
            for lanes in self._calendar.values()
            for lane in lanes
            for reservation in lane
        }
        return _ResourceTimelineCheckpoint(
            timeline_identity=id(self),
            resources=tuple(self._resources.items()),
            ready=tuple(self._ready.items()),
            calendar=tuple(
                (
                    resource_id,
                    tuple(tuple(lane) for lane in lanes),
                )
                for resource_id, lanes in self._calendar.items()
            ),
            tokens=tuple(
                (token, tuple(rows))
                for token, rows in self._tokens.items()
            ),
            reservation_ends=tuple(
                (reservation, reservation.end_us)
                for reservation in reservations.values()
            ),
            next_token=self._next_token,
        )

    def restore(self, checkpoint: object) -> None:
        if (
            not isinstance(checkpoint, _ResourceTimelineCheckpoint)
            or checkpoint.timeline_identity != id(self)
        ):
            raise SchedulerError("resource timeline checkpoint is invalid")
        for reservation, end_us in checkpoint.reservation_ends:
            reservation.end_us = end_us
        self._resources = dict(checkpoint.resources)
        self._ready = dict(checkpoint.ready)
        self._calendar = {
            resource_id: [list(lane) for lane in lanes]
            for resource_id, lanes in checkpoint.calendar
        }
        self._tokens = {
            token: list(rows) for token, rows in checkpoint.tokens
        }
        self._next_token = checkpoint.next_token

    def require_compatible(
        self, resources: Mapping[str, ResourceProfile]
    ) -> None:
        for resource_id, resource in resources.items():
            current = self._resources.get(resource_id)
            if current is None:
                raise SchedulerError(
                    f"shared timeline lacks resource: {resource_id}"
                )
            if current != resource:
                raise SchedulerError(
                    f"shared timeline resource differs: {resource_id}"
                )

    @staticmethod
    def _overlaps(
        start_us: int,
        end_us: int,
        reservation: _CalendarReservation,
    ) -> bool:
        return (
            reservation.start_us < reservation.end_us
            and start_us < reservation.end_us
            and reservation.start_us < end_us
        )

    @staticmethod
    def _first_overlap(
        lane: Sequence[_CalendarReservation],
        start_us: int,
        end_us: int,
    ) -> _CalendarReservation | None:
        """Find a conflict in a non-overlapping, start-ordered lane."""
        # Released reservations remain as zero-length audit rows. Search by
        # start time, then skip those rows without hiding a live reservation
        # at the same timestamp.
        left = 0
        right = len(lane)
        while left < right:
            middle = (left + right) // 2
            if lane[middle].start_us < end_us:
                left = middle + 1
            else:
                right = middle
        for index in range(left - 1, -1, -1):
            reservation = lane[index]
            if reservation.start_us >= reservation.end_us:
                continue
            if reservation.end_us <= start_us:
                return None
            if reservation.start_us < end_us:
                return reservation
        return None

    def _lane_is_free(
        self,
        resource_id: str,
        lane: int,
        start_us: int,
        end_us: int,
        tentative: Mapping[int, Sequence[tuple[int, int]]],
    ) -> bool:
        if self._first_overlap(
            self._calendar[resource_id][lane], start_us, end_us
        ) is not None:
            return False
        return not any(
            start_us < other_end and other_start < end_us
            for other_start, other_end in tentative.get(lane, ())
        )

    def _validate_internal_capacity(
        self,
        resource_id: str,
        demands: Sequence[LeaseDemand],
    ) -> None:
        capacity = self._resources[resource_id].capacity
        events: list[tuple[int, int]] = []
        for demand in demands:
            if demand.slots > capacity:
                raise SchedulerError("resource lease exceeds resource capacity")
            events.append((demand.start_offset_us, demand.slots))
            events.append(
                (
                    demand.start_offset_us + demand.duration_upper_us,
                    -demand.slots,
                )
            )
        active = 0
        for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
            active += delta
            if active > capacity:
                raise SchedulerError(
                    "route lease concurrency exceeds resource capacity"
                )

    def _assign_resource_at(
        self,
        resource_id: str,
        demands: Sequence[LeaseDemand],
        route_start_us: int,
    ) -> tuple[LeasePlan, ...] | None:
        capacity = self._resources[resource_id].capacity
        ordered = sorted(
            demands,
            key=lambda demand: (
                demand.start_offset_us,
                -demand.duration_upper_us,
                demand.lease_id,
            ),
        )
        first_start_us = route_start_us + min(
            demand.start_offset_us for demand in ordered
        )
        last_end_us = route_start_us + max(
            demand.start_offset_us + demand.duration_upper_us
            for demand in ordered
        )
        calendar = self._calendar[resource_id]
        if not any(calendar) or all(
            self._first_overlap(lane, first_start_us, last_end_us) is None
            for lane in calendar
        ):
            available_at_us = [first_start_us] * capacity
            plans = []
            for demand in ordered:
                start_us = route_start_us + demand.start_offset_us
                reserved_until_us = start_us + demand.duration_upper_us
                lanes = tuple(itertools.islice((
                    lane for lane in range(capacity)
                    if available_at_us[lane] <= start_us
                ), demand.slots))
                if len(lanes) != demand.slots:
                    break
                for lane in lanes:
                    available_at_us[lane] = reserved_until_us
                plans.append(LeasePlan(
                    lease_id=demand.lease_id,
                    resource_id=resource_id,
                    lanes=lanes,
                    start_us=start_us,
                    predicted_end_us=start_us + demand.duration_us,
                    reserved_until_us=reserved_until_us,
                ))
            else:
                return tuple(sorted(
                    plans, key=lambda plan: (plan.start_us, plan.lease_id)
                ))
        if len(ordered) == 1:
            demand = ordered[0]
            start_us = route_start_us + demand.start_offset_us
            reserved_until_us = start_us + demand.duration_upper_us
            lanes = tuple(itertools.islice((
                lane for lane in range(capacity)
                if self._lane_is_free(
                    resource_id,
                    lane,
                    start_us,
                    reserved_until_us,
                    {},
                )
            ), demand.slots))
            if len(lanes) != demand.slots:
                return None
            return (LeasePlan(
                lease_id=demand.lease_id,
                resource_id=resource_id,
                lanes=lanes,
                start_us=start_us,
                predicted_end_us=start_us + demand.duration_us,
                reserved_until_us=reserved_until_us,
            ),)
        tentative: dict[int, list[tuple[int, int]]] = {
            lane: [] for lane in range(capacity)
        }
        plans: list[LeasePlan] = []
        attempts = 0

        def assign(index: int) -> bool:
            nonlocal attempts
            if index == len(ordered):
                return True
            demand = ordered[index]
            start_us = route_start_us + demand.start_offset_us
            reserved_until_us = start_us + demand.duration_upper_us
            available = tuple(lane for lane in range(capacity) if self._lane_is_free(
                resource_id, lane, start_us, reserved_until_us, tentative))
            for lanes in itertools.combinations(available, demand.slots):
                attempts += 1
                if attempts > 100_000:
                    raise SchedulerError("resource lease assignment search limit")
                if not all(
                    self._lane_is_free(
                        resource_id,
                        lane,
                        start_us,
                        reserved_until_us,
                        tentative,
                    )
                    for lane in lanes
                ):
                    continue
                for lane in lanes:
                    tentative[lane].append((start_us, reserved_until_us))
                plans.append(
                    LeasePlan(
                        lease_id=demand.lease_id,
                        resource_id=resource_id,
                        lanes=tuple(lanes),
                        start_us=start_us,
                        predicted_end_us=start_us + demand.duration_us,
                        reserved_until_us=reserved_until_us,
                    )
                )
                if assign(index + 1):
                    return True
                plans.pop()
                for lane in lanes:
                    tentative[lane].pop()
            return False

        if not assign(0):
            return None
        return tuple(sorted(plans, key=lambda plan: (plan.start_us, plan.lease_id)))

    def _next_resource_start(
        self,
        resource_id: str,
        demands: Sequence[LeaseDemand],
        route_start_us: int,
    ) -> int:
        candidates: list[int] = []
        for demand in demands:
            start_us = route_start_us + demand.start_offset_us
            end_us = start_us + demand.duration_upper_us
            for lane in self._calendar[resource_id]:
                reservation = self._first_overlap(
                    lane, start_us, end_us
                )
                if reservation is not None:
                    candidate = (
                        reservation.end_us - demand.start_offset_us
                    )
                    if candidate > route_start_us:
                        candidates.append(candidate)
        if not candidates:
            raise SchedulerError("cannot advance resource lease calendar")
        return min(candidates)

    def _preview_resource(
        self,
        resource_id: str,
        demands: Sequence[LeaseDemand],
        not_before_us: int,
    ) -> tuple[int, tuple[LeasePlan, ...]]:
        self._validate_internal_capacity(resource_id, demands)
        candidate = not_before_us
        for _ in range(100_000):
            plans = self._assign_resource_at(resource_id, demands, candidate)
            if plans is not None:
                return candidate, plans
            candidate = self._next_resource_start(
                resource_id, demands, candidate
            )
        raise SchedulerError("resource queue prediction did not converge")

    def preview_leases(
        self,
        demands: Sequence[LeaseDemand],
        arrival_us: int,
        service_us: int,
        service_upper_us: int,
    ) -> LeasePreview:
        _strict_int("lease preview arrival_us", arrival_us)
        _strict_int("lease preview service_us", service_us, 1)
        _strict_int("lease preview service_upper_us", service_upper_us, service_us)
        if not demands:
            raise SchedulerError("lease preview requires at least one lease")
        grouped: dict[str, list[LeaseDemand]] = {}
        lease_ids: set[str] = set()
        for demand in demands:
            if demand.lease_id in lease_ids:
                raise SchedulerError("duplicate predicted resource lease id")
            lease_ids.add(demand.lease_id)
            resource = self._resources.get(demand.resource_id)
            if resource is None:
                raise SchedulerError("resource lease references an unknown resource")
            if not self._ready[demand.resource_id]:
                raise SchedulerError(f"resource is not ready: {demand.resource_id}")
            _strict_int("predicted lease slots", demand.slots, 1)
            _strict_int("predicted lease start offset", demand.start_offset_us)
            _strict_int("predicted lease duration", demand.duration_us, 1)
            _strict_int(
                "predicted lease upper duration",
                demand.duration_upper_us,
                demand.duration_us,
            )
            grouped.setdefault(demand.resource_id, []).append(demand)

        route_start_us = arrival_us
        blockers: dict[str, int] = {}
        final_plans: tuple[LeasePlan, ...] = ()
        for _ in range(100_000):
            plans: list[LeasePlan] = []
            moved = False
            for resource_id in sorted(grouped):
                ready_start_us, resource_plans = self._preview_resource(
                    resource_id,
                    grouped[resource_id],
                    route_start_us,
                )
                if ready_start_us > route_start_us:
                    route_start_us = ready_start_us
                    blockers[resource_id] = max(
                        blockers.get(resource_id, 0),
                        route_start_us - arrival_us,
                    )
                    moved = True
                    break
                plans.extend(resource_plans)
            if not moved:
                final_plans = tuple(
                    sorted(
                        plans,
                        key=lambda plan: (
                            plan.start_us,
                            plan.resource_id,
                            plan.lease_id,
                        ),
                    )
                )
                break
        else:
            raise SchedulerError("cross-resource queue prediction did not converge")

        queue_by_resource = {
            resource_id: blockers.get(resource_id, 0)
            for resource_id in sorted(grouped)
        }
        return LeasePreview(
            start_us=route_start_us,
            finish_us=route_start_us + service_us,
            finish_upper_us=route_start_us + service_upper_us,
            plans=final_plans,
            queue_by_resource_us=queue_by_resource,
            blocking_resources=tuple(
                resource_id
                for resource_id, delay in queue_by_resource.items()
                if delay > 0
            ),
        )

    def preview(
        self, resource_slots: Mapping[str, int], arrival_us: int, duration_us: int
    ) -> tuple[int, int, Mapping[str, tuple[int, ...]]]:
        demands = tuple(
            LeaseDemand(
                lease_id=f"{resource_id}-legacy",
                resource_id=resource_id,
                slots=count,
                start_offset_us=0,
                duration_us=duration_us,
                duration_upper_us=duration_us,
            )
            for resource_id, count in sorted(resource_slots.items())
        )
        result = self.preview_leases(
            demands,
            arrival_us,
            duration_us,
            duration_us,
        )
        selected = {
            plan.resource_id: plan.lanes
            for plan in result.plans
        }
        return result.start_us, result.finish_us, selected

    def commit(
        self, selected: Mapping[str, tuple[int, ...]], finish_us: int
    ) -> None:
        _strict_int("legacy resource finish_us", finish_us)
        for resource_id, lanes in selected.items():
            for lane in lanes:
                if self._calendar[resource_id][lane]:
                    raise SchedulerError(
                        "legacy commit cannot follow interval reservations"
                    )
                self._calendar[resource_id][lane].append(
                    _CalendarReservation(
                        token=f"legacy:{resource_id}:{lane}",
                        owner_id="legacy",
                        lease_id=f"{resource_id}-legacy",
                        start_us=0,
                        end_us=finish_us,
                    )
                )

    def commit_leases(
        self,
        preview: LeasePreview,
        owner_id: str,
    ) -> tuple[LeaseRecord, ...]:
        _text("lease owner_id", owner_id)
        records: list[LeaseRecord] = []
        for plan in preview.plans:
            for lane in plan.lanes:
                if self._first_overlap(
                    self._calendar[plan.resource_id][lane],
                    plan.start_us,
                    plan.reserved_until_us,
                ) is not None:
                    raise SchedulerError("resource lease changed before commit")

        for plan in preview.plans:
            token = f"lease-{self._next_token}"
            self._next_token += 1
            reservations: list[_CalendarReservation] = []
            for lane in plan.lanes:
                reservation = _CalendarReservation(
                    token=token,
                    owner_id=owner_id,
                    lease_id=plan.lease_id,
                    start_us=plan.start_us,
                    end_us=plan.reserved_until_us,
                )
                self._calendar[plan.resource_id][lane].append(reservation)
                self._calendar[plan.resource_id][lane].sort(
                    key=lambda row: (row.start_us, row.end_us, row.token)
                )
                reservations.append(reservation)
            self._tokens[token] = reservations
            records.append(
                LeaseRecord(
                    token=token,
                    owner_id=owner_id,
                    lease_id=plan.lease_id,
                    resource_id=plan.resource_id,
                    lanes=plan.lanes,
                    start_us=plan.start_us,
                    predicted_end_us=plan.predicted_end_us,
                    reserved_until_us=plan.reserved_until_us,
                )
            )
        return tuple(records)

    def release(self, token: str, actual_end_us: int) -> None:
        token = _text("lease token", token)
        reservations = self._tokens.get(token)
        if reservations is None:
            raise SchedulerError("unknown lease token")
        _strict_int("lease actual_end_us", actual_end_us)
        if any(
            actual_end_us < reservation.start_us
            or actual_end_us > reservation.end_us
            for reservation in reservations
        ):
            raise SchedulerError("actual lease completion is outside reservation")
        for reservation in reservations:
            reservation.end_us = actual_end_us

    def extend(self, token: str, reserved_until_us: int) -> int:
        token = _text("lease token", token)
        reservations = self._tokens.get(token)
        if reservations is None:
            raise SchedulerError("unknown lease token")
        _strict_int("lease reserved_until_us", reserved_until_us)
        current_end_us = reservations[0].end_us
        if any(
            reservation.end_us != current_end_us
            for reservation in reservations
        ):
            raise SchedulerError("lease token has inconsistent lane ends")
        if reserved_until_us < current_end_us:
            raise SchedulerError("lease extension cannot shorten a reservation")
        if reserved_until_us == current_end_us:
            return current_end_us

        owned = {id(reservation) for reservation in reservations}
        for lanes in self._calendar.values():
            for lane in lanes:
                for reservation in lane:
                    if id(reservation) not in owned:
                        continue
                    if any(
                        other.token != token
                        and self._overlaps(
                            reservation.start_us,
                            reserved_until_us,
                            other,
                        )
                        for other in lane
                    ):
                        raise SchedulerError(
                            "lease extension overlaps committed work"
                        )
        for reservation in reservations:
            reservation.end_us = reserved_until_us
        return current_end_us

    def extend_many(
        self,
        tokens: Sequence[str],
        reserved_until_us: int,
        *,
        cancelled_owner_ids: Sequence[str] = (),
        cancellation_at_us: int = 0,
    ) -> tuple[Mapping[str, int], Mapping[str, tuple[str, ...]]]:
        """Atomically cancel queued owners and extend a composite lease."""
        token_rows = tuple(_text("lease token", token) for token in tokens)
        if not token_rows or len(token_rows) != len(set(token_rows)):
            raise SchedulerError("composite lease tokens must be unique")
        _strict_int("lease reserved_until_us", reserved_until_us)
        owners = tuple(
            _text("cancelled lease owner_id", owner_id)
            for owner_id in cancelled_owner_ids
        )
        if len(owners) != len(set(owners)):
            raise SchedulerError("cancelled lease owners must be unique")
        _strict_int("lease cancellation at_us", cancellation_at_us)

        selected: dict[str, list[_CalendarReservation]] = {}
        previous: dict[str, int] = {}
        for token in token_rows:
            reservations = self._tokens.get(token)
            if reservations is None:
                raise SchedulerError("unknown lease token")
            current_end_us = reservations[0].end_us
            if any(
                reservation.end_us != current_end_us
                for reservation in reservations
            ):
                raise SchedulerError("lease token has inconsistent lane ends")
            if reserved_until_us < current_end_us:
                raise SchedulerError(
                    "lease extension cannot shorten a reservation"
                )
            if reservations[0].owner_id in owners:
                raise SchedulerError(
                    "composite extension cannot cancel its own lease"
                )
            selected[token] = reservations
            previous[token] = current_end_us

        selected_tokens = set(token_rows)
        cancelled_owners = set(owners)
        for resource_lanes in self._calendar.values():
            for lane in resource_lanes:
                for reservation in lane:
                    if reservation.token not in selected_tokens:
                        continue
                    for other in lane:
                        if other is reservation:
                            continue
                        other_end_us = other.end_us
                        if other.token in selected_tokens:
                            other_end_us = reserved_until_us
                        elif other.owner_id in cancelled_owners:
                            other_end_us = max(
                                other.start_us, cancellation_at_us
                            )
                        if (
                            reservation.start_us < reserved_until_us
                            and other.start_us < other_end_us
                            and reservation.start_us < other_end_us
                            and other.start_us < reserved_until_us
                        ):
                            raise SchedulerError(
                                "composite lease extension overlaps committed work"
                            )

        cancelled: dict[str, list[str]] = {owner: [] for owner in owners}
        for token, reservations in sorted(self._tokens.items()):
            if not reservations:
                continue
            owner_id = reservations[0].owner_id
            if owner_id not in cancelled_owners:
                continue
            changed = False
            for reservation in reservations:
                if reservation.end_us <= cancellation_at_us:
                    continue
                reservation.end_us = max(
                    reservation.start_us, cancellation_at_us
                )
                changed = True
            if changed:
                cancelled[owner_id].append(token)
        for reservations in selected.values():
            for reservation in reservations:
                reservation.end_us = reserved_until_us
        return (
            dict(sorted(previous.items())),
            {
                owner_id: tuple(cancelled[owner_id])
                for owner_id in sorted(cancelled)
            },
        )

    def retime_many(
        self, windows: Mapping[str, tuple[int, int]], *,
        cancelled_owner_ids: Sequence[str] = (), cancellation_at_us: int = 0,
    ) -> tuple[Mapping[str, int], Mapping[str, tuple[str, ...]]]:
        """Atomically advance phase boundaries without changing lease identity."""
        owners = set(cancelled_owner_ids)
        for token, (start, end) in windows.items():
            if token not in self._tokens:
                raise SchedulerError("unknown lease token")
            _strict_int("phase lease start", start)
            _strict_int("phase lease end", end, start)
            if any(row.owner_id in owners for row in self._tokens[token]):
                raise SchedulerError("phase change cannot cancel its own lease")

        def window(row):
            if row.token in windows:
                return windows[row.token]
            if row.owner_id in owners:
                return row.start_us, min(row.end_us, max(row.start_us, cancellation_at_us))
            return row.start_us, row.end_us

        for lanes in self._calendar.values():
            for lane in lanes:
                for row in lane:
                    if row.token not in windows:
                        continue
                    start, end = window(row)
                    for other in lane:
                        left, right = window(other)
                        if (row is not other and start < end and left < right
                                and start < right and left < end):
                            raise SchedulerError("phase lease change overlaps committed work")
        previous = {token: self._tokens[token][0].end_us for token in windows}
        cancelled = {owner: self.cancel_owner(owner, cancellation_at_us) for owner in sorted(owners)}
        for token, (start, end) in windows.items():
            for row in self._tokens[token]:
                row.start_us, row.end_us = start, end
        for lanes in self._calendar.values():
            for lane in lanes:
                lane.sort(key=lambda row: (row.start_us, row.end_us, row.token))
        return previous, cancelled

    def reassign_owner(
        self,
        tokens: Sequence[str],
        *,
        expected_owner_id: str,
        owner_id: str,
    ) -> None:
        """Atomically transfer unchanged leases to a different owner."""
        token_rows = tuple(_text("lease token", token) for token in tokens)
        if not token_rows or len(token_rows) != len(set(token_rows)):
            raise SchedulerError("lease owner transfer tokens must be unique")
        expected_owner_id = _text(
            "expected lease owner_id", expected_owner_id
        )
        owner_id = _text("lease owner_id", owner_id)
        selected: list[_CalendarReservation] = []
        for token in token_rows:
            reservations = self._tokens.get(token)
            if (
                reservations is None
                or not reservations
                or any(
                    reservation.owner_id != expected_owner_id
                    for reservation in reservations
                )
            ):
                raise SchedulerError("lease owner transfer state differs")
            selected.extend(reservations)
        for reservation in selected:
            reservation.owner_id = owner_id

    def cancel_owner(self, owner_id: str, at_us: int) -> tuple[str, ...]:
        owner_id = _text("cancelled lease owner_id", owner_id)
        _strict_int("lease cancellation at_us", at_us)
        cancelled: list[str] = []
        for token, reservations in sorted(self._tokens.items()):
            if not reservations or reservations[0].owner_id != owner_id:
                continue
            changed = False
            for reservation in reservations:
                if reservation.end_us <= at_us:
                    continue
                new_end_us = max(reservation.start_us, at_us)
                if new_end_us >= reservation.end_us:
                    continue
                reservation.end_us = new_end_us
                changed = True
            if changed:
                cancelled.append(token)
        return tuple(cancelled)

    def revoke_resource(
        self,
        resource_id: str,
        at_us: int,
    ) -> tuple[str, ...]:
        resource_id = _text("revoked resource_id", resource_id)
        if resource_id not in self._resources:
            raise SchedulerError("cannot revoke an unknown resource")
        _strict_int("resource revocation at_us", at_us)
        self._ready[resource_id] = False
        affected: set[str] = set()
        for lane in self._calendar[resource_id]:
            for reservation in lane:
                if reservation.end_us <= at_us:
                    continue
                if reservation.owner_id != "legacy":
                    affected.add(reservation.owner_id)
                reservation.end_us = max(reservation.start_us, at_us)
        return tuple(sorted(affected))

    def restore_resource(self, resource_id: str) -> None:
        resource_id = _text("restored resource_id", resource_id)
        if resource_id not in self._resources:
            raise SchedulerError("cannot restore an unknown resource")
        self._ready[resource_id] = True

    def is_ready(self, resource_id: str) -> bool:
        resource_id = _text("queried resource_id", resource_id)
        if resource_id not in self._resources:
            raise SchedulerError("cannot query an unknown resource")
        return self._ready[resource_id]

    def owned_busy_until_us(
        self,
        resource_ids: Sequence[str],
        at_us: int,
    ) -> int | None:
        """Return the known release bound for scheduler-owned work."""
        rows = tuple(_text("queried resource_id", row) for row in resource_ids)
        if not rows:
            raise SchedulerError("resource busy query requires resources")
        _strict_int("resource busy query at_us", at_us)
        unknown = set(rows) - set(self._resources)
        if unknown:
            raise SchedulerError("cannot query an unknown resource")
        ends = [
            reservation.end_us
            for resource_id in rows
            for lane in self._calendar[resource_id]
            for reservation in lane
            if reservation.owner_id != "legacy" and reservation.end_us > at_us
        ]
        return None if not ends else max(ends)

    def next_available_us(
        self,
        resource_id: str,
        not_before_us: int,
        duration_us: int = 1,
        slots: int = 1,
    ) -> int | None:
        resource_id = _text("queried resource_id", resource_id)
        resource = self._resources.get(resource_id)
        if resource is None:
            raise SchedulerError("cannot query an unknown resource")
        _strict_int("resource availability not_before_us", not_before_us)
        _strict_int("resource availability duration_us", duration_us, 1)
        _strict_int("resource availability slots", slots, 1)
        if slots > resource.capacity:
            raise SchedulerError("resource availability exceeds capacity")
        if not self._ready[resource_id]:
            return None
        demand = LeaseDemand(
            lease_id=f"{resource_id}-availability-query",
            resource_id=resource_id,
            slots=slots,
            start_offset_us=0,
            duration_us=duration_us,
            duration_upper_us=duration_us,
        )
        start_us, _ = self._preview_resource(
            resource_id, (demand,), not_before_us
        )
        return start_us

    def resource_snapshot(
        self, at_us: int
    ) -> Mapping[str, Mapping[str, object]]:
        _strict_int("resource snapshot at_us", at_us)
        result: dict[str, Mapping[str, object]] = {}
        for resource_id, resource in sorted(self._resources.items()):
            reservations = [
                reservation
                for lane in self._calendar[resource_id]
                for reservation in lane
                if reservation.end_us > at_us
            ]
            active = [
                reservation
                for reservation in reservations
                if reservation.start_us <= at_us < reservation.end_us
            ]
            free_slots = sum(
                not any(
                    reservation.start_us <= at_us < reservation.end_us
                    for reservation in lane
                )
                for lane in self._calendar[resource_id]
            )
            result[resource_id] = {
                "ready": self._ready[resource_id],
                "capacity": resource.capacity,
                "free_slots": free_slots if self._ready[resource_id] else 0,
                "next_free_us": (
                    at_us
                    if self._ready[resource_id] and free_slots > 0
                    else self.next_available_us(resource_id, at_us)
                ),
                "active_until_us": max(
                    (reservation.end_us for reservation in active),
                    default=at_us,
                ),
                "reserved_until_us": max(
                    (reservation.end_us for reservation in reservations),
                    default=at_us,
                ),
                "active_owners": sorted({
                    reservation.owner_id for reservation in active
                }),
                "queued_owners": sorted({
                    reservation.owner_id
                    for reservation in reservations
                    if reservation.start_us > at_us
                }),
            }
        return result

    def causal_state(self) -> Mapping[str, object]:
        """Return the complete state that can affect a future lease commit."""
        return {
            "next_token": self._next_token,
            "resources": {
                resource_id: {
                    "capacity": resource.capacity,
                    "lanes": [
                        [
                            {
                                "end_us": reservation.end_us,
                                "lease_id": reservation.lease_id,
                                "owner_id": reservation.owner_id,
                                "start_us": reservation.start_us,
                                "token": reservation.token,
                            }
                            for reservation in lane
                        ]
                        for lane in self._calendar[resource_id]
                    ],
                    "ready": self._ready[resource_id],
                }
                for resource_id, resource in sorted(
                    self._resources.items()
                )
            },
            "schema": "research-scheduler-resource-timeline-state-v1",
        }
