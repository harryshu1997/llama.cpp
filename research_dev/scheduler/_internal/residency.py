#!/usr/bin/env python3
"""Model-residency and parallel-route scheduling primitives.

This layer plans model cohorts on an exclusive accelerator and can assign a
subset of one cohort to an already-resident parallel route. It does not load
models or dispatch requests.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
from typing import Mapping, Sequence


RESIDENCY_PROBLEM_SCHEMA = "s42-residency-problem-v1"

__all__ = [
    "CohortRoute",
    "ParallelAssignment",
    "QueueRequest",
    "ResidencyPhase",
    "ResidencyResource",
    "ResidencySchedulerError",
    "ResidencySequence",
    "ResidencySwitch",
    "RESIDENCY_PROBLEM_SCHEMA",
    "RouteSchedule",
    "ScheduledRequest",
    "optimize_parallel_assignment",
    "parallel_assignment_to_json",
    "plan_residency_sequence",
    "residency_sequence_to_json",
    "schedule_route",
    "scheduled_request_to_json",
    "solve_residency_problem",
]


class ResidencySchedulerError(ValueError):
    pass


def _non_negative(name: str, value: int) -> int:
    if type(value) is not int or value < 0:
        raise ResidencySchedulerError(f"{name} must be an integer >= 0")
    return value


def _positive(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ResidencySchedulerError(f"{name} must be an integer > 0")
    return value


def _text(name: str, value: str) -> str:
    if type(value) is not str or not value:
        raise ResidencySchedulerError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ResidencySchedulerError(f"{name} must be ASCII") from exc
    return value


@dataclass(frozen=True)
class QueueRequest:
    request_id: str
    model_id: str
    arrival_us: int
    deadline_us: int
    priority: int = 0

    def validate(self) -> None:
        _text("request_id", self.request_id)
        _text("model_id", self.model_id)
        _non_negative("arrival_us", self.arrival_us)
        _positive("deadline_us", self.deadline_us)
        _non_negative("priority", self.priority)
        if self.deadline_us <= self.arrival_us:
            raise ResidencySchedulerError("deadline must follow arrival")


@dataclass(frozen=True)
class ResidencyResource:
    resource_id: str
    capacity_bytes: int
    allocation_limit_bytes: int = 0

    def validate(self) -> None:
        _text("resource_id", self.resource_id)
        _positive("capacity_bytes", self.capacity_bytes)
        _non_negative("allocation_limit_bytes", self.allocation_limit_bytes)
        if (
            self.allocation_limit_bytes
            and self.allocation_limit_bytes > self.capacity_bytes
        ):
            raise ResidencySchedulerError(
                "allocation limit exceeds resource capacity"
            )


@dataclass(frozen=True)
class CohortRoute:
    route_id: str
    model_id: str
    resource_id: str
    slots: int
    service_us: Mapping[str, int]
    residency_id: str
    resident_bytes: int
    preloaded: bool
    energy_uj: Mapping[str, int] | None = None

    def validate(
        self,
        requests: Sequence[QueueRequest],
        resource: ResidencyResource,
    ) -> None:
        _text("route_id", self.route_id)
        _text("route model_id", self.model_id)
        _text("route resource_id", self.resource_id)
        _text("route residency_id", self.residency_id)
        _positive("route slots", self.slots)
        _positive("route resident_bytes", self.resident_bytes)
        if self.resource_id != resource.resource_id:
            raise ResidencySchedulerError("route and resource do not match")
        if self.resident_bytes > resource.capacity_bytes:
            raise ResidencySchedulerError("route exceeds resource capacity")
        if (
            resource.allocation_limit_bytes
            and self.resident_bytes > resource.allocation_limit_bytes
        ):
            raise ResidencySchedulerError(
                "route exceeds resource allocation limit"
            )
        request_ids = {request.request_id for request in requests}
        if any(request.model_id != self.model_id for request in requests):
            raise ResidencySchedulerError("route and request model mismatch")
        if set(self.service_us) != request_ids:
            raise ResidencySchedulerError(
                "route service profile does not cover its requests"
            )
        for request_id, service_us in self.service_us.items():
            _text("service request id", request_id)
            _positive("service_us", service_us)
        if self.energy_uj is not None:
            if set(self.energy_uj) != request_ids:
                raise ResidencySchedulerError(
                    "route energy profile does not cover its requests"
                )
            for energy_uj in self.energy_uj.values():
                _positive("energy_uj", energy_uj)


@dataclass(frozen=True)
class ResidencySwitch:
    resource_id: str
    source_residency_id: str
    target_residency_id: str
    latency_us: int
    energy_uj: int | None
    measured: bool
    evidence_ids: tuple[str, ...]

    def validate(self) -> None:
        _text("switch resource_id", self.resource_id)
        _text("switch source_residency_id", self.source_residency_id)
        _text("switch target_residency_id", self.target_residency_id)
        _positive("switch latency_us", self.latency_us)
        if self.energy_uj is not None:
            _non_negative("switch energy_uj", self.energy_uj)
        if not self.evidence_ids:
            raise ResidencySchedulerError("switch has no evidence")
        for evidence_id in self.evidence_ids:
            _text("switch evidence id", evidence_id)


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    model_id: str
    route_id: str
    lane: int
    start_us: int
    finish_us: int
    deadline_met: bool


@dataclass(frozen=True)
class RouteSchedule:
    route_id: str
    ready_us: int
    jobs: tuple[ScheduledRequest, ...]
    finish_us: int
    energy_uj: int | None


@dataclass(frozen=True)
class ParallelAssignment:
    source_route_id: str
    target_route_id: str
    source_request_ids: tuple[str, ...]
    target_request_ids: tuple[str, ...]
    source_schedule: RouteSchedule
    target_schedule: RouteSchedule
    deadline_misses: int
    weighted_tardiness_us: int
    makespan_us: int
    weighted_completion_us: int
    energy_uj: int | None

    @property
    def objective(self) -> tuple[int, ...]:
        return (
            self.deadline_misses,
            self.weighted_tardiness_us,
            self.makespan_us,
            self.weighted_completion_us,
            self.energy_uj if self.energy_uj is not None else 0,
            len(self.source_request_ids),
        )


@dataclass(frozen=True)
class ResidencyPhase:
    model_id: str
    route_id: str
    residency_id: str
    prior_residency_id: str
    switch_start_us: int
    ready_us: int
    finish_us: int
    switch_latency_us: int
    jobs: tuple[ScheduledRequest, ...]


@dataclass(frozen=True)
class ResidencySequence:
    resource_id: str
    initial_residency_id: str
    model_order: tuple[str, ...]
    phases: tuple[ResidencyPhase, ...]
    deadline_misses: int
    weighted_tardiness_us: int
    makespan_us: int
    weighted_completion_us: int
    energy_uj: int | None
    measured: bool
    evidence_ids: tuple[str, ...]

    @property
    def objective(self) -> tuple[int, ...]:
        return (
            self.deadline_misses,
            self.weighted_tardiness_us,
            self.makespan_us,
            self.weighted_completion_us,
            int(self.energy_uj is None),
            self.energy_uj if self.energy_uj is not None else 0,
        )


def _priority_weight(priority: int) -> int:
    return 10 ** min(priority, 6)


def schedule_route(
    requests: Sequence[QueueRequest],
    route: CohortRoute,
    resource: ResidencyResource,
    ready_us: int,
) -> RouteSchedule:
    _non_negative("route ready_us", ready_us)
    for request in requests:
        request.validate()
    route.validate(requests, resource)
    lanes = [(ready_us, lane) for lane in range(route.slots)]
    heapq.heapify(lanes)
    jobs: list[ScheduledRequest] = []
    for request in sorted(
        requests, key=lambda item: (item.arrival_us, item.request_id)
    ):
        free_us, lane = heapq.heappop(lanes)
        start_us = max(ready_us, free_us, request.arrival_us)
        finish_us = start_us + route.service_us[request.request_id]
        heapq.heappush(lanes, (finish_us, lane))
        jobs.append(ScheduledRequest(
            request_id=request.request_id,
            model_id=request.model_id,
            route_id=route.route_id,
            lane=lane,
            start_us=start_us,
            finish_us=finish_us,
            deadline_met=finish_us <= request.deadline_us,
        ))
    energy = (
        None
        if route.energy_uj is None
        else sum(route.energy_uj[request.request_id] for request in requests)
    )
    return RouteSchedule(
        route_id=route.route_id,
        ready_us=ready_us,
        jobs=tuple(jobs),
        finish_us=max((job.finish_us for job in jobs), default=ready_us),
        energy_uj=energy,
    )


def _assignment_metrics(
    requests: Sequence[QueueRequest],
    schedules: Sequence[RouteSchedule],
) -> tuple[int, int, int, int, int | None]:
    by_request = {
        job.request_id: job
        for schedule in schedules
        for job in schedule.jobs
    }
    if set(by_request) != {request.request_id for request in requests}:
        raise ResidencySchedulerError("assignment lost or duplicated requests")
    deadline_misses = 0
    tardiness = 0
    completion = 0
    for request in requests:
        job = by_request[request.request_id]
        weight = _priority_weight(request.priority)
        late = max(0, job.finish_us - request.deadline_us)
        deadline_misses += int(late > 0)
        tardiness += weight * late
        completion += weight * (job.finish_us - request.arrival_us)
    energies = [schedule.energy_uj for schedule in schedules]
    energy = None if any(value is None for value in energies) else sum(energies)  # type: ignore[arg-type]
    return (
        deadline_misses,
        tardiness,
        max((job.finish_us for job in by_request.values()), default=0),
        completion,
        energy,
    )


def optimize_parallel_assignment(
    requests: Sequence[QueueRequest],
    source_route: CohortRoute,
    source_resource: ResidencyResource,
    source_ready_us: int,
    target_route: CohortRoute,
    target_resource: ResidencyResource,
    target_ready_us: int,
    source_eligible_ids: frozenset[str] | None = None,
    makespan_first: bool = False,
) -> ParallelAssignment:
    """Find an exact request split between two routes for one model."""
    if not requests:
        raise ResidencySchedulerError("parallel assignment has no requests")
    if len(requests) > 24:
        raise ResidencySchedulerError(
            "exact parallel assignment is limited to 24 requests"
        )
    request_ids = {request.request_id for request in requests}
    eligible = request_ids if source_eligible_ids is None else set(source_eligible_ids)
    if not eligible <= request_ids:
        raise ResidencySchedulerError("source eligibility references an unknown request")
    source_route.validate(requests, source_resource)
    target_route.validate(requests, target_resource)
    if source_route.model_id != target_route.model_id:
        raise ResidencySchedulerError("parallel routes serve different models")

    best: tuple[tuple[int, ...], ParallelAssignment] | None = None
    ordered = list(requests)
    for mask in range(1 << len(ordered)):
        source_requests = [
            request for index, request in enumerate(ordered)
            if mask & (1 << index)
        ]
        if any(request.request_id not in eligible for request in source_requests):
            continue
        target_requests = [
            request for index, request in enumerate(ordered)
            if not mask & (1 << index)
        ]
        source_schedule = schedule_route(
            source_requests,
            CohortRoute(
                **{
                    **source_route.__dict__,
                    "service_us": {
                        request.request_id: source_route.service_us[request.request_id]
                        for request in source_requests
                    },
                    "energy_uj": (
                        None
                        if source_route.energy_uj is None
                        else {
                            request.request_id: source_route.energy_uj[request.request_id]
                            for request in source_requests
                        }
                    ),
                }
            ),
            source_resource,
            source_ready_us,
        )
        target_schedule = schedule_route(
            target_requests,
            CohortRoute(
                **{
                    **target_route.__dict__,
                    "service_us": {
                        request.request_id: target_route.service_us[request.request_id]
                        for request in target_requests
                    },
                    "energy_uj": (
                        None
                        if target_route.energy_uj is None
                        else {
                            request.request_id: target_route.energy_uj[request.request_id]
                            for request in target_requests
                        }
                    ),
                }
            ),
            target_resource,
            target_ready_us,
        )
        metrics = _assignment_metrics(
            requests, (source_schedule, target_schedule)
        )
        candidate = ParallelAssignment(
            source_route_id=source_route.route_id,
            target_route_id=target_route.route_id,
            source_request_ids=tuple(
                request.request_id for request in source_requests
            ),
            target_request_ids=tuple(
                request.request_id for request in target_requests
            ),
            source_schedule=source_schedule,
            target_schedule=target_schedule,
            deadline_misses=metrics[0],
            weighted_tardiness_us=metrics[1],
            makespan_us=metrics[2],
            weighted_completion_us=metrics[3],
            energy_uj=metrics[4],
        )
        key = (
            (
                candidate.makespan_us,
                candidate.deadline_misses,
                candidate.weighted_tardiness_us,
                candidate.weighted_completion_us,
                candidate.energy_uj if candidate.energy_uj is not None else 0,
                len(candidate.source_request_ids),
            )
            if makespan_first
            else candidate.objective
        )
        if best is None or key < best[0]:
            best = (key, candidate)
    if best is None:
        raise ResidencySchedulerError("parallel assignment search is empty")
    return best[1]


def _switch_map(
    switches: Sequence[ResidencySwitch],
    resource: ResidencyResource,
    require_measured: bool,
) -> Mapping[tuple[str, str], ResidencySwitch]:
    result: dict[tuple[str, str], ResidencySwitch] = {}
    for switch in switches:
        switch.validate()
        if switch.resource_id != resource.resource_id:
            raise ResidencySchedulerError("switch and resource do not match")
        if require_measured and not switch.measured:
            continue
        key = (switch.source_residency_id, switch.target_residency_id)
        if key in result:
            raise ResidencySchedulerError("duplicate residency switch")
        result[key] = switch
    return result


def plan_residency_sequence(
    requests: Sequence[QueueRequest],
    routes: Sequence[CohortRoute],
    resource: ResidencyResource,
    initial_residency_id: str,
    initial_ready_us: int,
    switches: Sequence[ResidencySwitch],
    require_measured: bool = True,
) -> ResidencySequence:
    """Enumerate model-cohort orders on one exclusive residency resource."""
    if not requests or not routes:
        raise ResidencySchedulerError("residency sequence is empty")
    resource.validate()
    _text("initial residency id", initial_residency_id)
    _non_negative("initial ready_us", initial_ready_us)
    by_model: dict[str, list[QueueRequest]] = {}
    for request in requests:
        request.validate()
        by_model.setdefault(request.model_id, []).append(request)
    route_by_model: dict[str, CohortRoute] = {}
    for route in routes:
        if route.model_id in route_by_model:
            raise ResidencySchedulerError("multiple cohort routes for one model")
        model_requests = by_model.get(route.model_id)
        if not model_requests:
            raise ResidencySchedulerError("route has no model requests")
        route.validate(model_requests, resource)
        route_by_model[route.model_id] = route
    if set(route_by_model) != set(by_model):
        raise ResidencySchedulerError("model route coverage is incomplete")
    switch_by_pair = _switch_map(switches, resource, require_measured)

    best: ResidencySequence | None = None
    for order in itertools.permutations(sorted(by_model)):
        now_us = initial_ready_us
        residency_id = initial_residency_id
        phases: list[ResidencyPhase] = []
        measured = True
        evidence: set[str] = set()
        total_switch_energy: int | None = 0
        feasible = True
        for model_id in order:
            route = route_by_model[model_id]
            prior = residency_id
            if residency_id == route.residency_id:
                switch_latency = 0
                switch_start = now_us
            else:
                switch = switch_by_pair.get((residency_id, route.residency_id))
                if switch is None:
                    feasible = False
                    break
                switch_start = now_us
                switch_latency = switch.latency_us
                now_us += switch_latency
                if switch.energy_uj is None:
                    total_switch_energy = None
                elif total_switch_energy is not None:
                    total_switch_energy += switch.energy_uj
                measured = measured and switch.measured
                evidence.update(switch.evidence_ids)
                residency_id = route.residency_id
            schedule = schedule_route(
                by_model[model_id], route, resource, now_us
            )
            phases.append(ResidencyPhase(
                model_id=model_id,
                route_id=route.route_id,
                residency_id=route.residency_id,
                prior_residency_id=prior,
                switch_start_us=switch_start,
                ready_us=now_us,
                finish_us=schedule.finish_us,
                switch_latency_us=switch_latency,
                jobs=schedule.jobs,
            ))
            now_us = schedule.finish_us
        if not feasible:
            continue
        schedules = tuple(
            RouteSchedule(
                route_id=phase.route_id,
                ready_us=phase.ready_us,
                jobs=phase.jobs,
                finish_us=phase.finish_us,
                energy_uj=route_by_model[phase.model_id].energy_uj and sum(
                    route_by_model[phase.model_id].energy_uj[job.request_id]
                    for job in phase.jobs
                ),
            )
            for phase in phases
        )
        metrics = _assignment_metrics(requests, schedules)
        energy = (
            None
            if metrics[4] is None or total_switch_energy is None
            else metrics[4] + total_switch_energy
        )
        candidate = ResidencySequence(
            resource_id=resource.resource_id,
            initial_residency_id=initial_residency_id,
            model_order=tuple(order),
            phases=tuple(phases),
            deadline_misses=metrics[0],
            weighted_tardiness_us=metrics[1],
            makespan_us=metrics[2],
            weighted_completion_us=metrics[3],
            energy_uj=energy,
            measured=measured,
            evidence_ids=tuple(sorted(evidence)),
        )
        if best is None or candidate.objective < best.objective:
            best = candidate
    if best is None:
        raise ResidencySchedulerError(
            "no model order has qualified residency transitions"
        )
    return best


def solve_residency_problem(value: object) -> ResidencySequence:
    if type(value) is not dict or value.get("schema") != RESIDENCY_PROBLEM_SCHEMA:
        raise ResidencySchedulerError("residency problem schema mismatch")
    try:
        raw_resource = value["resource"]
        raw_requests = value["requests"]
        raw_routes = value["routes"]
        raw_switches = value["switches"]
        if (
            type(raw_resource) is not dict
            or type(raw_requests) is not list
            or type(raw_routes) is not list
            or type(raw_switches) is not list
        ):
            raise ResidencySchedulerError("residency problem fields are invalid")
        resource = ResidencyResource(
            resource_id=raw_resource["resource_id"],
            capacity_bytes=raw_resource["capacity_bytes"],
            allocation_limit_bytes=raw_resource.get(
                "allocation_limit_bytes", 0
            ),
        )
        requests = tuple(
            QueueRequest(
                request_id=row["request_id"],
                model_id=row["model_id"],
                arrival_us=row["arrival_us"],
                deadline_us=row["deadline_us"],
                priority=row.get("priority", 0),
            )
            for row in raw_requests
        )
        routes = tuple(
            CohortRoute(
                route_id=row["route_id"],
                model_id=row["model_id"],
                resource_id=resource.resource_id,
                slots=row["slots"],
                service_us=row["service_us"],
                residency_id=row["residency_id"],
                resident_bytes=row["resident_bytes"],
                preloaded=row["preloaded"],
                energy_uj=row.get("energy_uj"),
            )
            for row in raw_routes
        )
        switches = tuple(
            ResidencySwitch(
                resource_id=resource.resource_id,
                source_residency_id=row["source_residency_id"],
                target_residency_id=row["target_residency_id"],
                latency_us=row["latency_us"],
                energy_uj=row["energy_uj"],
                measured=row["measured"],
                evidence_ids=tuple(row["evidence_ids"]),
            )
            for row in raw_switches
        )
        return plan_residency_sequence(
            requests=requests,
            routes=routes,
            resource=resource,
            initial_residency_id=value["initial_residency_id"],
            initial_ready_us=value["initial_ready_us"],
            switches=switches,
            require_measured=value.get("require_measured", True),
        )
    except (KeyError, TypeError) as exc:
        raise ResidencySchedulerError(
            "residency problem fields are invalid"
        ) from exc


def scheduled_request_to_json(job: ScheduledRequest) -> dict[str, object]:
    return {
        "request_id": job.request_id,
        "model_id": job.model_id,
        "route_id": job.route_id,
        "lane": job.lane,
        "start_us": job.start_us,
        "finish_us": job.finish_us,
        "deadline_met": job.deadline_met,
    }


def parallel_assignment_to_json(
    plan: ParallelAssignment,
) -> dict[str, object]:
    def schedule(value: RouteSchedule) -> dict[str, object]:
        return {
            "route_id": value.route_id,
            "ready_us": value.ready_us,
            "finish_us": value.finish_us,
            "energy_uj": value.energy_uj,
            "jobs": [scheduled_request_to_json(job) for job in value.jobs],
        }

    return {
        "source_route_id": plan.source_route_id,
        "target_route_id": plan.target_route_id,
        "source_request_ids": list(plan.source_request_ids),
        "target_request_ids": list(plan.target_request_ids),
        "source_schedule": schedule(plan.source_schedule),
        "target_schedule": schedule(plan.target_schedule),
        "deadline_misses": plan.deadline_misses,
        "weighted_tardiness_us": plan.weighted_tardiness_us,
        "makespan_us": plan.makespan_us,
        "weighted_completion_us": plan.weighted_completion_us,
        "energy_uj": plan.energy_uj,
    }


def residency_sequence_to_json(
    plan: ResidencySequence,
) -> dict[str, object]:
    return {
        "resource_id": plan.resource_id,
        "initial_residency_id": plan.initial_residency_id,
        "model_order": list(plan.model_order),
        "deadline_misses": plan.deadline_misses,
        "weighted_tardiness_us": plan.weighted_tardiness_us,
        "makespan_us": plan.makespan_us,
        "weighted_completion_us": plan.weighted_completion_us,
        "energy_uj": plan.energy_uj,
        "measured": plan.measured,
        "evidence_ids": list(plan.evidence_ids),
        "phases": [
            {
                "model_id": phase.model_id,
                "route_id": phase.route_id,
                "residency_id": phase.residency_id,
                "prior_residency_id": phase.prior_residency_id,
                "switch_start_us": phase.switch_start_us,
                "ready_us": phase.ready_us,
                "finish_us": phase.finish_us,
                "switch_latency_us": phase.switch_latency_us,
                "jobs": [
                    scheduled_request_to_json(job) for job in phase.jobs
                ],
            }
            for phase in plan.phases
        ],
    }
