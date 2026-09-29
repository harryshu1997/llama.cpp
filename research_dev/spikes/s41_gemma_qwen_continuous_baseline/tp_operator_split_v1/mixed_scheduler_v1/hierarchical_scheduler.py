#!/usr/bin/env python3
"""Hierarchical task, layer, and operator placement primitives."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Request:
    request_id: int
    model_id: str
    arrival_ms: float
    input_tokens: int
    output_tokens: int
    deadline_ms: float
    priority: int = 0


@dataclass(frozen=True)
class RouteJob:
    request: Request
    service_ms: float


@dataclass(frozen=True)
class RouteProfile:
    route_id: str
    model_id: str
    devices: frozenset[str]
    slots: int
    ready_ms: float
    load_ms: float = 0.0
    power_w: float = 0.0
    qualified: bool = True

    def validate(self) -> None:
        if self.slots <= 0:
            raise ValueError("route slots must be positive")
        if self.ready_ms < 0 or self.load_ms < 0 or self.power_w < 0:
            raise ValueError("route times and power must be non-negative")


@dataclass(frozen=True)
class ScheduledJob:
    request_id: int
    route_id: str
    start_ms: float
    completion_ms: float
    slot: int


@dataclass(frozen=True)
class AssignmentResult:
    source_request_ids: tuple[int, ...]
    target_request_ids: tuple[int, ...]
    source_jobs: tuple[ScheduledJob, ...]
    target_jobs: tuple[ScheduledJob, ...]
    makespan_ms: float
    deadline_misses: int
    weighted_tardiness_ms: float
    weighted_completion_ms: float
    energy_j: float

    @property
    def objective(self) -> tuple[float, ...]:
        return (
            float(self.deadline_misses),
            self.weighted_tardiness_ms,
            self.makespan_ms,
            self.weighted_completion_ms,
            self.energy_j,
        )


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    gpu_ready_ms: float
    source_only_makespan_ms: float
    promoted_makespan_ms: float
    saved_ms: float
    reason: str


@dataclass(frozen=True)
class LayerModel:
    model_id: str
    total_layers: int
    fixed_gpu_bytes: int
    gpu_bytes_per_layer: int
    cpu_ms_per_layer: float
    gpu_ms_per_layer: float
    boundary_ms: float


@dataclass(frozen=True)
class LayerPlan:
    gpu_layers: int
    cpu_layers: int
    gpu_bytes: int
    estimated_ms: float
    mode: str
    reason: str


@dataclass(frozen=True)
class OperatorShape:
    model_id: str
    family: str
    m: int
    n: int
    k: int
    weight_format: str


@dataclass(frozen=True)
class OperatorPlan:
    backend: str
    split_axis: str
    split_amount: int
    qualified: bool
    reason: str


GEMMA4_12B_FFN_POLICY = (
    (1, 9664),
    (2, 8192),
    (4, 6144),
    (128, 8192),
    (512, 11136),
)


def _priority_weight(priority: int) -> float:
    return float(10 ** max(0, min(priority, 6)))


def schedule_jobs(
    profile: RouteProfile,
    jobs: Iterable[RouteJob],
    order: str = "fifo",
) -> tuple[ScheduledJob, ...]:
    """Schedule non-preemptive jobs on a measured llama-server route."""
    profile.validate()
    pending = list(jobs)
    if any(job.request.model_id != profile.model_id for job in pending):
        raise ValueError("route and request model mismatch")
    if any(job.service_ms < 0 for job in pending):
        raise ValueError("negative service time")
    if order == "fifo":
        pending.sort(key=lambda job: (
            job.request.arrival_ms, job.request.request_id
        ))
    elif order == "lpt":
        pending.sort(key=lambda job: (
            max(profile.ready_ms, job.request.arrival_ms),
            -job.service_ms,
            job.request.request_id,
        ))
    elif order == "edf":
        pending.sort(key=lambda job: (
            max(profile.ready_ms, job.request.arrival_ms),
            job.request.deadline_ms,
            -job.request.priority,
            job.request.request_id,
        ))
    else:
        raise ValueError(f"unknown order: {order}")

    slots = [(profile.ready_ms, slot) for slot in range(profile.slots)]
    heapq.heapify(slots)
    result = []
    for job in pending:
        free_ms, slot = heapq.heappop(slots)
        start_ms = max(free_ms, profile.ready_ms, job.request.arrival_ms)
        completion_ms = start_ms + job.service_ms
        heapq.heappush(slots, (completion_ms, slot))
        result.append(ScheduledJob(
            request_id=job.request.request_id,
            route_id=profile.route_id,
            start_ms=start_ms,
            completion_ms=completion_ms,
            slot=slot,
        ))
    return tuple(result)


def summarize_assignment(
    source_profile: RouteProfile,
    target_profile: RouteProfile,
    source_jobs: Sequence[RouteJob],
    target_jobs: Sequence[RouteJob],
    target_order: str,
) -> AssignmentResult:
    source_scheduled = schedule_jobs(source_profile, source_jobs, "fifo")
    target_scheduled = schedule_jobs(target_profile, target_jobs, target_order)
    by_request = {
        job.request.request_id: job.request
        for job in itertools.chain(source_jobs, target_jobs)
    }
    scheduled = source_scheduled + target_scheduled
    completion_values = [job.completion_ms for job in scheduled]
    makespan_ms = max(
        [source_profile.ready_ms, target_profile.ready_ms, *completion_values]
    )
    deadline_misses = 0
    weighted_tardiness_ms = 0.0
    weighted_completion_ms = 0.0
    for job in scheduled:
        request = by_request[job.request_id]
        weight = _priority_weight(request.priority)
        tardiness = max(0.0, job.completion_ms - request.deadline_ms)
        deadline_misses += int(tardiness > 0)
        weighted_tardiness_ms += weight * tardiness
        weighted_completion_ms += weight * (
            job.completion_ms - request.arrival_ms
        )
    source_busy_ms = sum(job.service_ms for job in source_jobs)
    target_busy_ms = sum(job.service_ms for job in target_jobs)
    energy_j = (
        source_busy_ms * source_profile.power_w
        + target_busy_ms * target_profile.power_w
    ) / 1000
    return AssignmentResult(
        source_request_ids=tuple(
            job.request.request_id for job in source_jobs
        ),
        target_request_ids=tuple(
            job.request.request_id for job in target_jobs
        ),
        source_jobs=source_scheduled,
        target_jobs=target_scheduled,
        makespan_ms=makespan_ms,
        deadline_misses=deadline_misses,
        weighted_tardiness_ms=weighted_tardiness_ms,
        weighted_completion_ms=weighted_completion_ms,
        energy_j=energy_j,
    )


def optimize_task_assignment(
    requests: Sequence[Request],
    source_profile: RouteProfile,
    target_profile: RouteProfile,
    source_service_ms: dict[int, float],
    target_service_ms: dict[int, float],
    target_order: str = "lpt",
    makespan_first: bool = False,
) -> AssignmentResult:
    """Find the exact task-level split for at most 24 queued requests."""
    if len(requests) > 24:
        raise ValueError("exact assignment is limited to 24 requests")
    ids = {request.request_id for request in requests}
    if set(source_service_ms) != ids or set(target_service_ms) != ids:
        raise ValueError("service profile does not cover every request")
    if source_profile.model_id != target_profile.model_id:
        raise ValueError("task routes must serve the same model")

    best = None
    request_list = list(requests)
    for mask in range(1 << len(request_list)):
        source = []
        target = []
        for index, request in enumerate(request_list):
            if mask & (1 << index):
                source.append(RouteJob(
                    request, source_service_ms[request.request_id]
                ))
            else:
                target.append(RouteJob(
                    request, target_service_ms[request.request_id]
                ))
        candidate = summarize_assignment(
            source_profile, target_profile, source, target, target_order
        )
        if makespan_first:
            key = (
                candidate.makespan_ms,
                candidate.deadline_misses,
                candidate.weighted_tardiness_ms,
                candidate.weighted_completion_ms,
                candidate.energy_j,
            )
        else:
            key = candidate.objective
        if best is None or key < best[0]:
            best = (key, candidate)
    if best is None:
        raise ValueError("empty assignment search")
    return best[1]


def decide_promotion(
    now_ms: float,
    protected_release_ms: float,
    next_protected_arrival_ms: float | None,
    switch_back_ms: float,
    guard_ms: float,
    min_gain_ms: float,
    source_only_makespan_ms: float,
    promoted_makespan_ms: float,
    target_load_ms: float,
) -> PromotionDecision:
    gpu_ready_ms = max(now_ms, protected_release_ms) + target_load_ms
    protected_horizon_ms = gpu_ready_ms + switch_back_ms + guard_ms
    if (
        next_protected_arrival_ms is not None
        and next_protected_arrival_ms < protected_horizon_ms
    ):
        return PromotionDecision(
            False,
            gpu_ready_ms,
            source_only_makespan_ms,
            promoted_makespan_ms,
            source_only_makespan_ms - promoted_makespan_ms,
            "next protected request arrives before switch-back guard",
        )
    saved_ms = source_only_makespan_ms - promoted_makespan_ms
    if saved_ms < min_gain_ms:
        return PromotionDecision(
            False,
            gpu_ready_ms,
            source_only_makespan_ms,
            promoted_makespan_ms,
            saved_ms,
            "predicted gain is below promotion hysteresis",
        )
    return PromotionDecision(
        True,
        gpu_ready_ms,
        source_only_makespan_ms,
        promoted_makespan_ms,
        saved_ms,
        "GPU idle interval amortizes model load and switch-back guard",
    )


def plan_layers(
    model: LayerModel,
    free_vram_bytes: int,
    reserve_vram_bytes: int,
    protected_gpu: bool,
    allowed_protected_slowdown: float,
    measured_protected_slowdown_per_layer: float | None,
) -> LayerPlan:
    """Choose a contiguous GPU suffix for a route created at model load."""
    if model.total_layers <= 0 or model.gpu_bytes_per_layer <= 0:
        raise ValueError("invalid layer model")
    budget = max(0, free_vram_bytes - reserve_vram_bytes)
    if budget < model.fixed_gpu_bytes:
        max_gpu_layers = 0
    else:
        max_gpu_layers = min(
            model.total_layers,
            (budget - model.fixed_gpu_bytes) // model.gpu_bytes_per_layer,
        )
    best = None
    rejected_for_qos = False
    for gpu_layers in range(max_gpu_layers + 1):
        if protected_gpu and gpu_layers:
            if measured_protected_slowdown_per_layer is None:
                rejected_for_qos = True
                continue
            slowdown = gpu_layers * measured_protected_slowdown_per_layer
            if slowdown > allowed_protected_slowdown:
                rejected_for_qos = True
                continue
        cpu_layers = model.total_layers - gpu_layers
        estimated_ms = (
            cpu_layers * model.cpu_ms_per_layer
            + gpu_layers * model.gpu_ms_per_layer
            + (model.boundary_ms if 0 < gpu_layers < model.total_layers else 0)
        )
        key = (estimated_ms, -gpu_layers)
        if best is None or key < best[0]:
            best = (key, gpu_layers, cpu_layers, estimated_ms)
    if best is None:
        best = (
            (model.total_layers * model.cpu_ms_per_layer, 0),
            0,
            model.total_layers,
            model.total_layers * model.cpu_ms_per_layer,
        )
    _, gpu_layers, cpu_layers, estimated_ms = best
    gpu_bytes = (
        model.fixed_gpu_bytes + gpu_layers * model.gpu_bytes_per_layer
        if gpu_layers else 0
    )
    if gpu_layers == 0:
        mode = "cpu"
    elif gpu_layers == model.total_layers:
        mode = "full_gpu"
    else:
        mode = "partial_gpu"
    if rejected_for_qos and gpu_layers == 0:
        reason = "GPU layer route lacks a qualified protected-workload slowdown"
    elif max_gpu_layers == 0:
        reason = "VRAM reserve leaves no capacity for a GPU layer"
    else:
        reason = "lowest qualified layer-route latency within the VRAM budget"
    return LayerPlan(
        gpu_layers=gpu_layers,
        cpu_layers=cpu_layers,
        gpu_bytes=gpu_bytes,
        estimated_ms=estimated_ms,
        mode=mode,
        reason=reason,
    )


def _policy_value(table: Sequence[tuple[int, int]], m: int) -> int:
    for limit, value in table:
        if m <= limit:
            return value
    return 0


def plan_operator(
    shape: OperatorShape,
    parent_route: str,
    phone_weight_bytes_free: int,
    gpu_staging_profiled: bool,
) -> OperatorPlan:
    """Select only physically qualified operator-level routes."""
    if min(shape.m, shape.n, shape.k) <= 0:
        raise ValueError("operator dimensions must be positive")
    if parent_route == "gemma_cpu_op15":
        if (
            shape.model_id == "gemma-4-12b-it-q4_0"
            and shape.family == "gated_ffn"
            and shape.weight_format == "Q4_0"
            and shape.k == 3840
            and shape.n == 15360
        ):
            columns = _policy_value(GEMMA4_12B_FFN_POLICY, shape.m)
            if columns == 0:
                return OperatorPlan(
                    "cpu", "none", 0, True,
                    "M exceeds the qualified OP15 continuous-batch table",
                )
            required = 48 * 3 * columns * shape.k * 9 // 16
            if required > phone_weight_bytes_free:
                return OperatorPlan(
                    "cpu", "none", 0, True,
                    "resident HTP weight budget is insufficient",
                )
            return OperatorPlan(
                "cpu+op15_htp",
                "ffn_intermediate_columns",
                columns,
                True,
                "exact Gemma Q4_0 FFN shape uses the measured DMA-BUF table",
            )
        if shape.family in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            return OperatorPlan(
                "cpu", "none", 0, False,
                "attention projections lack a qualified integrated HTP route",
            )
        if shape.family == "lm_head":
            return OperatorPlan(
                "cpu", "none", 0, False,
                "Gemma Q4_0 lm_head is not resident or qualified on HTP",
            )
        return OperatorPlan(
            "cpu", "none", 0, True,
            "operator is dependency-bound or below the phone offload floor",
        )
    if parent_route in {"qwen_cuda_full", "gemma_cuda_full"}:
        if not gpu_staging_profiled:
            return OperatorPlan(
                "cuda", "none", 0, True,
                "VRAM input has no qualified CUDA-to-phone staging path",
            )
        return OperatorPlan(
            "cuda", "none", 0, False,
            "phone split loses to the measured resident CUDA route",
        )
    return OperatorPlan(
        "cpu", "none", 0, False,
        "parent route has no calibrated operator policy",
    )


def lower_bound_makespan_ms(
    jobs: Sequence[RouteJob], profile: RouteProfile
) -> float:
    if not jobs:
        return profile.ready_ms
    work_bound = sum(job.service_ms for job in jobs) / profile.slots
    longest = max(job.service_ms for job in jobs)
    first_arrival = min(job.request.arrival_ms for job in jobs)
    return max(profile.ready_ms, first_arrival) + max(work_bound, longest)


def optimality_gap(
    measured_makespan_ms: float,
    lower_bound_ms: float,
) -> tuple[float, float]:
    if lower_bound_ms <= 0 or measured_makespan_ms < lower_bound_ms:
        raise ValueError("invalid optimality bound")
    gap_ms = measured_makespan_ms - lower_bound_ms
    return gap_ms, 100.0 * gap_ms / lower_bound_ms
