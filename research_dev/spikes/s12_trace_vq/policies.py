#!/usr/bin/env python3
"""Frozen policy primitives for the S12 bounded virtual-queue replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from s12lib import S12Error, require_int


POLICIES = (
    "server_only_optimized",
    "causal_server_batch",
    "fixed_phone",
    "memory_admission_triggered",
)


def largest_supported_batch(queue_length: int, supported_batches: tuple[int, ...]) -> int | None:
    require_int("queue_length", queue_length)
    choices = [batch for batch in supported_batches if batch <= queue_length]
    return choices[-1] if choices else None


def causal_batch_size(
    queue_length: int,
    oldest_ready: bool,
    supported_batches: tuple[int, ...],
) -> int | None:
    """Causal: this function sees only current queue state, never future arrivals."""
    if not oldest_ready and queue_length < supported_batches[-1]:
        return None
    return largest_supported_batch(queue_length, supported_batches)


@dataclass(frozen=True)
class MemoryChoice:
    route: str | None
    batch_size: int | None
    server_admission_feasible: bool
    reason: str


def memory_admission_choice(
    queue_length: int,
    oldest_ready: bool,
    supported_batches: tuple[int, ...],
    available_hbm_mib: int,
    rows: dict[int, dict[str, Any]],
    phone_ready: bool,
) -> MemoryChoice:
    """Choose phone only if no currently dispatchable server batch fits HBM."""
    requested = causal_batch_size(queue_length, oldest_ready, supported_batches)
    if requested is None:
        return MemoryChoice(None, None, False, "BATCH_HOLD")

    for batch in reversed([b for b in supported_batches if b <= requested]):
        if rows[batch]["control_ready_hbm_mib"] <= available_hbm_mib:
            return MemoryChoice("SERVER_ONLY", batch, True, "SERVER_ADMISSION_FEASIBLE")

    if not phone_ready:
        return MemoryChoice(None, None, False, "SERVER_INFEASIBLE_PHONE_NOT_READY")
    for batch in reversed([b for b in supported_batches if b <= requested]):
        if rows[batch]["phone_route_ready_hbm_mib"] <= available_hbm_mib:
            return MemoryChoice("A0_OP15", batch, False, "SERVER_ADMISSION_INFEASIBLE")
    return MemoryChoice(None, None, False, "NO_ROUTE_FITS_HBM")


def background_hbm_at(t_us: int, timeline: list[dict[str, int]]) -> int:
    require_int("t_us", t_us)
    used = timeline[0]["used_mib"]
    for point in timeline:
        if point["t_us"] > t_us:
            break
        used = point["used_mib"]
    return used


def interval_fits_hbm(
    start_us: int,
    duration_us: int,
    route_hbm_mib: int,
    capacity_mib: int,
    timeline: list[dict[str, int]],
) -> bool:
    end_us = start_us + duration_us
    check_times = [start_us]
    check_times.extend(
        point["t_us"] for point in timeline if start_us < point["t_us"] < end_us
    )
    return all(
        background_hbm_at(t_us, timeline) + route_hbm_mib <= capacity_mib
        for t_us in check_times
    )


def earliest_hbm_start(
    earliest_us: int,
    duration_us: int,
    route_hbm_mib: int,
    capacity_mib: int,
    timeline: list[dict[str, int]],
    horizon_us: int,
) -> int | None:
    candidates = [earliest_us]
    candidates.extend(point["t_us"] for point in timeline if point["t_us"] >= earliest_us)
    for candidate in sorted(set(candidates)):
        if candidate + duration_us > horizon_us:
            continue
        if interval_fits_hbm(
            candidate,
            duration_us,
            route_hbm_mib,
            capacity_mib,
            timeline,
        ):
            return candidate
    return None


def _partitions(total: int, batches: tuple[int, ...]):
    current: list[int] = []

    def visit(remaining: int):
        if remaining == 0:
            yield tuple(current)
            return
        for batch in batches:
            if batch <= remaining:
                current.append(batch)
                yield from visit(remaining - batch)
                current.pop()

    yield from visit(total)


def offline_server_plan(
    arrivals: list[int],
    rows: dict[int, dict[str, Any]],
    capacity_mib: int,
    timeline: list[dict[str, int]],
    queue_limit: int,
    horizon_us: int,
    max_requests: int,
) -> list[dict[str, int]]:
    """Exhaustive clairvoyant FCFS partition reference for small traces.

    This is intentionally non-causal. It reads every future arrival and chooses
    the complete partition with lexicographic objective:
    (makespan, sum completion time, batch count, batch tuple).
    """
    n_requests = len(arrivals)
    if n_requests > max_requests:
        raise S12Error(
            f"server_only_optimized: {n_requests} requests exceed offline_max_requests={max_requests}"
        )
    supported = tuple(sorted(rows))
    best_objective = None
    best_groups = None
    for partition in _partitions(n_requests, supported):
        groups = []
        starts = [0] * n_requests
        completions = [0] * n_requests
        cursor = 0
        previous_finish = 0
        feasible = True
        for batch in partition:
            last = cursor + batch - 1
            row = rows[batch]
            earliest = max(previous_finish, arrivals[last])
            start = earliest_hbm_start(
                earliest,
                row["server_group_us"],
                row["control_ready_hbm_mib"],
                capacity_mib,
                timeline,
                horizon_us,
            )
            if start is None:
                feasible = False
                break
            finish = start + row["server_group_us"]
            if finish > horizon_us:
                feasible = False
                break
            for index in range(cursor, cursor + batch):
                starts[index] = start
                completions[index] = finish
            groups.append(
                {
                    "batch_size": batch,
                    "finish_us": finish,
                    "start_us": start,
                }
            )
            cursor += batch
            previous_finish = finish
        if not feasible:
            continue
        maximum_waiting = 0
        for now in arrivals:
            waiting = sum(
                1
                for index, arrival in enumerate(arrivals)
                if arrival <= now and starts[index] > now
            )
            maximum_waiting = max(maximum_waiting, waiting)
        if maximum_waiting > queue_limit:
            continue
        objective = (
            groups[-1]["finish_us"] if groups else 0,
            sum(completions),
            len(groups),
            partition,
        )
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_groups = groups
    if best_groups is None:
        raise S12Error("server_only_optimized: no feasible finite-queue schedule")
    return best_groups
