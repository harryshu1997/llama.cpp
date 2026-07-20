#!/usr/bin/env python3
"""Deterministic S14 fast-path policy primitives.

This module contains no device I/O and makes no performance or energy claim.
Measured profile rows are inputs. The live runtime may use a decision only after
binding it to current READY, lease, boundary, and device epochs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


class PolicyError(ValueError):
    pass


def _int(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PolicyError(f"{name} must be an integer >= {minimum}")
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise PolicyError(f"{name} must be a non-empty string")
    return value


def _evidence_id(name: str, value: object) -> str:
    text = _text(name, value)
    digest, separator, claim = text.partition("#")
    if separator != "#" or not claim or not digest.startswith("sha256:") \
            or len(digest) != 71 \
            or any(char not in "0123456789abcdef" for char in digest[7:]):
        raise PolicyError(f"{name} must be sha256:<64-hex>#<claim>")
    return text


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    service_class: str
    model_id: str
    island_id: str
    compatibility_key: str
    arrival_us: int
    deadline_us: int
    priority_class: int

    def validate(self, now_us: int) -> None:
        _text("request_id", self.request_id)
        _text("service_class", self.service_class)
        _text("model_id", self.model_id)
        _text("island_id", self.island_id)
        _text("compatibility_key", self.compatibility_key)
        _int("arrival_us", self.arrival_us)
        _int("deadline_us", self.deadline_us)
        _int("priority_class", self.priority_class)
        if self.deadline_us < self.arrival_us:
            raise PolicyError("deadline_us precedes arrival_us")
        if self.arrival_us > now_us:
            raise PolicyError("future work cannot enter the READY queue")


@dataclass(frozen=True)
class BatchPoint:
    batch_size: int
    duration_us: int

    def validate(self) -> None:
        _int("batch_size", self.batch_size, 1)
        _int("duration_us", self.duration_us, 1)


@dataclass(frozen=True)
class CertifiedBatchPoint(BatchPoint):
    correctness_certificate_id: str
    placement_certificate_id: str

    def validate(self) -> None:
        super().validate()
        _evidence_id("correctness_certificate_id", self.correctness_certificate_id)
        _evidence_id("placement_certificate_id", self.placement_certificate_id)


@dataclass(frozen=True)
class BatchDecision:
    action: str
    compatibility_key: str | None
    request_ids: tuple[str, ...]
    batch_size: int
    duration_us: int
    next_wake_us: int | None
    reason: str


@dataclass(frozen=True)
class BoundaryCertificate:
    request_id: str
    identity_ok: bool
    epoch_ok: bool
    correctness_ok: bool
    d2h_complete: bool

    def admitted(self) -> bool:
        _text("request_id", self.request_id)
        for name, value in (
            ("identity_ok", self.identity_ok),
            ("epoch_ok", self.epoch_ok),
            ("correctness_ok", self.correctness_ok),
            ("d2h_complete", self.d2h_complete),
        ):
            if type(value) is not bool:
                raise PolicyError(f"{name} must be bool")
        return self.identity_ok and self.epoch_ok and self.correctness_ok and self.d2h_complete


@dataclass(frozen=True)
class OperatingPlan:
    label: str
    selected_gpu_uuid: str
    offered_work: int
    completed_work: int
    slo_misses: int
    selected_gpu_energy_nj: int
    second_gpu_work: int

    def validate(self, expected_gpu_uuid: str, offered_work: int) -> None:
        if self.label not in {"P0", "P1", "P2", "P3"}:
            raise PolicyError(f"unexpected operating-plan label {self.label!r}")
        if self.selected_gpu_uuid != expected_gpu_uuid:
            raise PolicyError(f"{self.label}: selected GPU UUID mismatch")
        _int(f"{self.label}.offered_work", self.offered_work, 1)
        _int(f"{self.label}.completed_work", self.completed_work)
        _int(f"{self.label}.slo_misses", self.slo_misses)
        _int(f"{self.label}.selected_gpu_energy_nj", self.selected_gpu_energy_nj, 1)
        _int(f"{self.label}.second_gpu_work", self.second_gpu_work)
        if self.offered_work != offered_work:
            raise PolicyError(f"{self.label}: offered work differs from the frozen cohort")
        if self.completed_work > self.offered_work:
            raise PolicyError(f"{self.label}: completed work exceeds offered work")
        if self.second_gpu_work != 0:
            raise PolicyError(f"{self.label}: second GPU performed experiment work")


def _profile(points: Sequence[BatchPoint]) -> tuple[BatchPoint, ...]:
    if not points:
        raise PolicyError("batch profile is empty")
    ordered = tuple(sorted(points, key=lambda point: point.batch_size))
    seen: set[int] = set()
    for point in ordered:
        point.validate()
        if point.batch_size in seen:
            raise PolicyError("duplicate batch size")
        seen.add(point.batch_size)
    if ordered[0].batch_size != 1:
        raise PolicyError("batch profile must include batch size 1")
    return ordered


def throughput_knee(
    points: Sequence[BatchPoint],
    knee_num: int = 95,
    knee_den: int = 100,
) -> int:
    """Return the smallest batch reaching a rational fraction of peak throughput."""
    ordered = _profile(points)
    _int("knee_num", knee_num, 1)
    _int("knee_den", knee_den, 1)
    if knee_num > knee_den:
        raise PolicyError("knee fraction exceeds one")
    peak = ordered[0]
    for point in ordered[1:]:
        if point.batch_size * peak.duration_us > peak.batch_size * point.duration_us:
            peak = point
    for point in ordered:
        # point throughput >= knee_num / knee_den * peak throughput
        if (
            point.batch_size * peak.duration_us * knee_den
            >= knee_num * peak.batch_size * point.duration_us
        ):
            return point.batch_size
    raise AssertionError("peak point did not satisfy its own knee")


def _ordered_group(items: Iterable[WorkItem]) -> list[WorkItem]:
    return sorted(
        items,
        key=lambda item: (
            item.priority_class,
            item.deadline_us,
            item.arrival_us,
            item.request_id,
        ),
    )


def _feasible(
    now_us: int,
    items: Sequence[WorkItem],
    point: BatchPoint,
) -> bool:
    finish_us = now_us + point.duration_us
    return all(finish_us <= item.deadline_us for item in items[: point.batch_size])


def choose_batch(
    now_us: int,
    ready: Sequence[WorkItem],
    points: Sequence[BatchPoint],
    roofline_class: str,
    high_priority_max: int = 0,
    knee_num: int = 95,
    knee_den: int = 100,
) -> BatchDecision:
    """Choose one compatible native batch or a bounded wait.

    Smaller priority_class values are more urgent. Work with different
    compatibility keys is never placed in one batch.
    """
    _int("now_us", now_us)
    _int("high_priority_max", high_priority_max)
    if roofline_class not in {"memory_bound", "compute_bound"}:
        raise PolicyError("roofline_class must be memory_bound or compute_bound")
    ordered_points = _profile(points)
    if not ready:
        return BatchDecision("NO_WORK", None, (), 0, 0, None, "ready_queue_empty")
    for item in ready:
        item.validate(now_us)

    groups: dict[str, list[WorkItem]] = {}
    for item in ready:
        groups.setdefault(item.compatibility_key, []).append(item)

    for key, items in groups.items():
        signatures = {
            (item.service_class, item.model_id, item.island_id)
            for item in items
        }
        if len(signatures) != 1:
            raise PolicyError(f"compatibility key {key!r} aliases incompatible work")

    decisions: list[tuple[tuple[int, int, str], BatchDecision]] = []

    for key in sorted(groups):
        items = _ordered_group(groups[key])
        rank = (items[0].priority_class, items[0].deadline_us, key)
        available = [point for point in ordered_points if point.batch_size <= len(items)]
        feasible = [point for point in available if _feasible(now_us, items, point)]
        if not feasible:
            decisions.append((rank, BatchDecision(
                "NO_FEASIBLE", key, (), 0, 0, None, "deadline_infeasible"
            )))
            continue

        urgent = items[0].priority_class <= high_priority_max
        if roofline_class == "memory_bound":
            target = ordered_points[-1]
            current = feasible[-1]
        else:
            knee = throughput_knee(ordered_points, knee_num, knee_den)
            target = next(point for point in ordered_points if point.batch_size == knee)
            at_or_below_knee = [point for point in feasible if point.batch_size <= knee]
            current = at_or_below_knee[-1] if at_or_below_knee else feasible[0]

        target_ready = len(items) >= target.batch_size and _feasible(now_us, items, target)
        if urgent or target_ready:
            point = target if target_ready and not urgent else current
            selected = tuple(item.request_id for item in items[: point.batch_size])
            reason = "urgent_bypass" if urgent and not target_ready else "target_batch_ready"
            decision = BatchDecision(
                "LAUNCH", key, selected, point.batch_size, point.duration_us, None, reason
            )
            decisions.append((rank, decision))
            continue

        target_deadline = min(item.deadline_us for item in items)
        target_latest = target_deadline - target.duration_us
        current_deadline = min(item.deadline_us for item in items[: current.batch_size])
        current_latest = current_deadline - current.duration_us
        wake = max(now_us, min(target_latest, current_latest))
        if wake <= now_us:
            selected = tuple(item.request_id for item in items[: current.batch_size])
            decision = BatchDecision(
                "LAUNCH",
                key,
                selected,
                current.batch_size,
                current.duration_us,
                None,
                "latest_start_reached",
            )
            decisions.append((rank, decision))
        else:
            decisions.append((rank, BatchDecision(
                "WAIT", key, (), 0, 0, wake, "bounded_batch_wait"
            )))

    if not decisions:
        raise AssertionError("non-empty READY queue produced no decision")
    return min(decisions, key=lambda item: item[0])[1]


def select_operating_plan(
    plans: Sequence[OperatingPlan],
    expected_gpu_uuid: str,
    offered_work: int,
) -> OperatingPlan:
    """Select minimum selected-GPU energy among equal-work, zero-SLO-miss plans."""
    _text("expected_gpu_uuid", expected_gpu_uuid)
    _int("offered_work", offered_work, 1)
    by_label: dict[str, OperatingPlan] = {}
    for plan in plans:
        plan.validate(expected_gpu_uuid, offered_work)
        if plan.label in by_label:
            raise PolicyError(f"duplicate operating-plan label {plan.label}")
        by_label[plan.label] = plan
    if set(by_label) != {"P0", "P1", "P2", "P3"}:
        raise PolicyError("operating plans must contain exactly P0, P1, P2, and P3")
    eligible = [
        plan
        for plan in by_label.values()
        if plan.completed_work == offered_work and plan.slo_misses == 0
    ]
    if not eligible:
        raise PolicyError("no equal-work SLO-valid operating plan")
    return min(eligible, key=lambda plan: (plan.selected_gpu_energy_nj, plan.label))
