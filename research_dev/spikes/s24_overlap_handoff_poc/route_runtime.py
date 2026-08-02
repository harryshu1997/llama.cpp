#!/usr/bin/env python3
"""Typed finite-route runtime for the S24 fixed-diamond proof."""

from __future__ import annotations

import math
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
S23 = HERE.parent / "s23_dense_trace_runtime"
for dependency in (S23, S22):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from async_pipeline import DeviceBatcher as S22DeviceBatcher
from runtime_support import SequenceSlotPool, SerializedStageClient
from stage_v3_client import (
    BatchResult,
    BatchRow,
    Hello,
    ProtocolError,
    STAGE_V3_CAP_TERMINAL,
)


class RouteError(RuntimeError):
    pass


class RouteCleanupError(RouteError):
    def __init__(
        self,
        primary: BaseException | None,
        cleanup_errors: Sequence[BaseException],
    ):
        self.primary = primary
        self.cleanup_errors = tuple(cleanup_errors)
        message = f"route cleanup failed in {len(self.cleanup_errors)} operation(s)"
        if primary is not None:
            message += f" after {type(primary).__name__}"
        super().__init__(message)


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class StageEndpoint:
    host: str
    port: int

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("endpoint host must be nonempty")
        if not 1 <= self.port <= 65535:
            raise ValueError("endpoint port is out of range")


@dataclass(frozen=True)
class RowContribution:
    route_id: str
    upstream_worker: str
    stage_name: str
    priority: int
    request_id: int
    route_epoch: int
    seq_id: int
    position: int
    enqueued_ns: int
    latest_safe_ns: int
    gather_deadline_ns: int


def _row_key(row: BatchRow) -> tuple[int, int, int, int]:
    return row.request_id, row.route_epoch, row.seq_id, row.position


class _RecordingStageClient:
    def __init__(self, owner: "RouteDeviceBatcher", client: SerializedStageClient):
        self._owner = owner
        self._client = client

    def batch(self, rows: Sequence[BatchRow]) -> tuple[BatchResult, ...]:
        contributions = self._owner._take_contributions(rows)
        dispatch_ns = time.monotonic_ns()
        reason = self._owner._dispatch_reason(contributions, dispatch_ns)
        event = {
            "worker": self._owner.name,
            "batch_size": len(rows),
            "dispatch_reason": reason,
            "dispatch_ns": dispatch_ns,
            "request_ids": [row.request_id for row in rows],
            "route_epochs": [row.route_epoch for row in rows],
            "seq_ids": [row.seq_id for row in rows],
            "positions": [row.position for row in rows],
            "priorities": [item.priority for item in contributions],
            "routes": [item.route_id for item in contributions],
            "contributing_routes": sorted({
                item.route_id for item in contributions
            }),
            "upstream_workers": [
                item.upstream_worker for item in contributions
            ],
            "contributing_upstreams": sorted({
                item.upstream_worker for item in contributions
            }),
            "max_queue_us": max(
                (dispatch_ns - item.enqueued_ns) // 1000
                for item in contributions
            ),
            "latest_safe_ns": min(
                item.latest_safe_ns for item in contributions
            ),
            "gather_deadline_ns": min(
                item.gather_deadline_ns for item in contributions
            ),
        }
        compute_start_ns = time.monotonic_ns()
        try:
            results = self._client.batch(rows)
        except BaseException as exc:
            compute_end_ns = time.monotonic_ns()
            event.update({
                "compute_start_ns": compute_start_ns,
                "compute_end_ns": compute_end_ns,
                "compute_us": (compute_end_ns - compute_start_ns) // 1000,
                "status": "ERROR",
                "error_type": type(exc).__name__,
            })
            self._owner._append_event(event)
            raise
        compute_end_ns = time.monotonic_ns()
        expected = tuple(_row_key(row) for row in rows)
        actual = tuple(
            (row.request_id, row.route_epoch, row.seq_id, row.position)
            for row in results
        )
        if actual != expected:
            event.update({
                "compute_start_ns": compute_start_ns,
                "compute_end_ns": compute_end_ns,
                "compute_us": (compute_end_ns - compute_start_ns) // 1000,
                "status": "ERROR",
                "error_type": "ProtocolError",
                "error": "result lineage mismatch",
            })
            self._owner._append_event(event)
            raise ProtocolError(f"{self._owner.name} result lineage mismatch")
        event.update({
            "compute_start_ns": compute_start_ns,
            "compute_end_ns": compute_end_ns,
            "compute_us": (compute_end_ns - compute_start_ns) // 1000,
            "status": "OK",
        })
        self._owner._append_event(event)
        return results


class RouteDeviceBatcher:
    """Add route lineage and dispatch evidence to the S22 DeviceBatcher."""

    def __init__(
        self,
        name: str,
        client: SerializedStageClient,
        batch_knee: int,
        gather_us: int,
        queue_depth: int,
    ):
        if not name:
            raise ValueError("batcher name must be nonempty")
        _positive_int("batch_knee", batch_knee)
        if gather_us < 0:
            raise ValueError("gather_us cannot be negative")
        _positive_int("queue_depth", queue_depth)
        self.name = name
        self.client = client
        self.batch_knee = batch_knee
        self.gather_us = gather_us
        self.events: list[dict] = []
        self._contributions: dict[
            tuple[int, int, int, int], RowContribution
        ] = {}
        self._lock = threading.Lock()
        self._recording_client = _RecordingStageClient(self, client)
        self._batcher = S22DeviceBatcher(
            name,
            self._recording_client,
            batch_knee,
            gather_us,
            queue_depth,
        )

    def submit(
        self,
        row: BatchRow,
        route_id: str,
        upstream_worker: str,
        timeout_s: float,
        batch_wait_us: int | None,
        priority: int = 0,
    ) -> Future[BatchResult]:
        if not route_id or not upstream_worker:
            raise ValueError("route and upstream identifiers must be nonempty")
        if timeout_s <= 0:
            raise ValueError("timeout must be positive")
        if batch_wait_us is not None and batch_wait_us < 0:
            raise ValueError("batch wait cannot be negative")
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
            raise ValueError("priority must be a nonnegative integer")
        enqueued_ns = time.monotonic_ns()
        effective_wait_us = self.gather_us if batch_wait_us is None else min(
            self.gather_us, batch_wait_us,
        )
        contribution = RowContribution(
            route_id=route_id,
            upstream_worker=upstream_worker,
            stage_name=self.name,
            priority=priority,
            request_id=row.request_id,
            route_epoch=row.route_epoch,
            seq_id=row.seq_id,
            position=row.position,
            enqueued_ns=enqueued_ns,
            latest_safe_ns=enqueued_ns + effective_wait_us * 1000,
            gather_deadline_ns=enqueued_ns + self.gather_us * 1000,
        )
        key = _row_key(row)
        with self._lock:
            if key in self._contributions:
                raise RouteError(f"{self.name} duplicate pending row")
            self._contributions[key] = contribution
        try:
            future = self._batcher.submit(
                row, timeout_s, batch_wait_us, priority,
            )
        except BaseException:
            self._forget_contribution(key)
            raise
        future.add_done_callback(
            lambda _future, row_key=key: self._forget_contribution(row_key)
        )
        return future

    def _take_contributions(
        self, rows: Sequence[BatchRow],
    ) -> tuple[RowContribution, ...]:
        with self._lock:
            missing = [
                _row_key(row) for row in rows
                if _row_key(row) not in self._contributions
            ]
            if missing:
                raise RouteError(f"{self.name} missing row contribution")
            return tuple(
                self._contributions.pop(_row_key(row)) for row in rows
            )

    def _forget_contribution(self, key: tuple[int, int, int, int]) -> None:
        with self._lock:
            self._contributions.pop(key, None)

    def _dispatch_reason(
        self,
        contributions: Sequence[RowContribution],
        dispatch_ns: int,
    ) -> str:
        if len(contributions) >= self.batch_knee:
            return "BATCH_KNEE"
        earliest_safe = min(item.latest_safe_ns for item in contributions)
        earliest_gather = min(
            item.gather_deadline_ns for item in contributions
        )
        if earliest_safe < earliest_gather and dispatch_ns >= earliest_safe:
            return "LATEST_SAFE_START"
        return "GATHER_TIMER"

    def _append_event(self, event: dict) -> None:
        with self._lock:
            self.events.append(event)

    def pending(self) -> int:
        with self._lock:
            return len(self._contributions)

    def stop(self, timeout_s: float) -> None:
        self._batcher.stop(timeout_s)
        if self.pending():
            raise RouteError(f"{self.name} stopped with pending row metadata")


@dataclass(frozen=True)
class ResidentStage:
    worker_name: str
    layer_start: int
    layer_end: int
    endpoint: StageEndpoint
    client: SerializedStageClient
    slots: SequenceSlotPool
    batcher: RouteDeviceBatcher
    terminal: bool

    def __post_init__(self) -> None:
        if not self.worker_name:
            raise ValueError("worker name must be nonempty")
        if self.layer_start < 0 or self.layer_end <= self.layer_start:
            raise ValueError("stage layer range is invalid")
        if self.batcher.client is not self.client:
            raise ValueError("stage batcher and cleanup client must match")


@dataclass(frozen=True)
class FiniteRoute:
    route_id: str
    stages: tuple[ResidentStage, ...]

    def __post_init__(self) -> None:
        if not self.route_id or not self.stages:
            raise ValueError("route id and stages must be nonempty")
        if self.stages[0].layer_start != 0:
            raise ValueError("route must begin at layer zero")
        names = [stage.worker_name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("route repeats a physical worker")
        for left, right in zip(self.stages, self.stages[1:]):
            if left.layer_end != right.layer_start:
                raise ValueError("route has a layer boundary gap or overlap")
            if left.terminal:
                raise ValueError("only the last route stage may be terminal")
        if not self.stages[-1].terminal:
            raise ValueError("route must end at a terminal stage")


@dataclass(frozen=True)
class RouteRequest:
    request_id: int
    route_epoch: int
    route_id: str
    prompt_tokens: tuple[int, ...]
    output_steps: int
    slo_us: int
    priority: int
    batch_wait_us: int | None = None
    prefill_chunk: int | None = None

    def __post_init__(self) -> None:
        _positive_int("request_id", self.request_id)
        _positive_int("route_epoch", self.route_epoch)
        _positive_int("output_steps", self.output_steps)
        _positive_int("slo_us", self.slo_us)
        if not self.route_id:
            raise ValueError("route id must be nonempty")
        if not self.prompt_tokens or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.prompt_tokens
        ):
            raise ValueError("prompt tokens must be nonnegative integers")
        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, int)
            or self.priority < 0
        ):
            raise ValueError("priority must be a nonnegative integer")
        if self.batch_wait_us is not None and self.batch_wait_us < 0:
            raise ValueError("batch wait cannot be negative")
        if self.prefill_chunk is not None:
            _positive_int("prefill_chunk", self.prefill_chunk)


@dataclass(frozen=True)
class BoundaryActivation:
    worker_name: str
    layer_end: int
    position: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class RouteOutcome:
    request_id: int
    route_epoch: int
    route_id: str
    priority: int
    prompt_length: int
    output_tokens: tuple[int, ...]
    boundaries: tuple[BoundaryActivation, ...]
    scheduled_arrival_ns: int
    call_ns: int
    lease_ready_ns: int
    first_token_ns: int
    completed_ns: int
    slo_us: int

    @property
    def ttft_us(self) -> int:
        return (self.first_token_ns - self.scheduled_arrival_ns) // 1000

    @property
    def latency_us(self) -> int:
        return (self.completed_ns - self.scheduled_arrival_ns) // 1000

    @property
    def lease_queue_us(self) -> int:
        return (self.lease_ready_ns - self.call_ns) // 1000

    @property
    def slo_met(self) -> bool:
        return self.latency_us <= self.slo_us


def validate_stage_hello(stage: ResidentStage, hello: Hello) -> None:
    if (
        hello.layer_start != stage.layer_start
        or hello.layer_end != stage.layer_end
    ):
        raise ProtocolError(f"{stage.worker_name} layer range mismatch")
    terminal = bool(hello.capabilities & STAGE_V3_CAP_TERMINAL)
    if terminal != stage.terminal:
        raise ProtocolError(f"{stage.worker_name} terminal capability mismatch")
    if hello.max_streams != stage.slots.capacity:
        raise ProtocolError(f"{stage.worker_name} lease capacity mismatch")
    if stage.batcher.batch_knee > min(hello.n_batch, hello.n_ubatch):
        raise ProtocolError(f"{stage.worker_name} batch knee exceeds capacity")


EXPECTED_FIXED_ROUTES = {
    "R0": (
        ("cuda-prefix", 0, 8),
        ("cuda-mid", 8, 16),
        ("cuda-tail", 16, 48),
    ),
    "R1": (
        ("cuda-prefix", 0, 8),
        ("op15-mid", 8, 16),
        ("cuda-tail", 16, 48),
    ),
    "R2": (
        ("op12-prefix", 0, 8),
        ("op15-mid", 8, 16),
        ("cuda-tail", 16, 48),
    ),
}


def validate_fixed_routes(routes: Sequence[FiniteRoute]) -> None:
    by_id = {route.route_id: route for route in routes}
    if len(by_id) != len(routes) or set(by_id) != set(EXPECTED_FIXED_ROUTES):
        raise RouteError("fixed diamond must define exactly R0, R1, and R2")
    for route_id, expected in EXPECTED_FIXED_ROUTES.items():
        actual = tuple(
            (stage.worker_name, stage.layer_start, stage.layer_end)
            for stage in by_id[route_id].stages
        )
        if actual != expected:
            raise RouteError(f"{route_id} differs from the fixed route")


def validate_shared_treatment(routes: Sequence[FiniteRoute]) -> None:
    validate_fixed_routes(routes)
    by_id = {route.route_id: route for route in routes}
    r0, r1, r2 = by_id["R0"], by_id["R1"], by_id["R2"]
    if r1.stages[1].batcher is not r2.stages[1].batcher:
        raise RouteError("R1 and R2 must share one OP15 batcher")
    tail_batchers = {
        id(route.stages[-1].batcher) for route in (r0, r1, r2)
    }
    if len(tail_batchers) != 1:
        raise RouteError("R0, R1, and R2 must share one CUDA-tail batcher")


class RouteRunner:
    def __init__(self, routes: Sequence[FiniteRoute]):
        if not routes:
            raise ValueError("at least one route is required")
        if len({route.route_id for route in routes}) != len(routes):
            raise ValueError("route ids must be unique")
        self._routes = {route.route_id: route for route in routes}
        self._pinned: dict[int, tuple[int, str]] = {}
        self._lock = threading.Lock()

    def pinned(self) -> dict[int, tuple[int, str]]:
        with self._lock:
            return dict(self._pinned)

    def run(
        self,
        request: RouteRequest,
        timeout_s: float,
        scheduled_arrival_ns: int | None = None,
        capture_boundaries: bool = True,
    ) -> RouteOutcome:
        if timeout_s <= 0:
            raise ValueError("timeout must be positive")
        route = self._routes.get(request.route_id)
        if route is None:
            raise RouteError(f"unknown route {request.route_id}")
        call_ns = time.monotonic_ns()
        if scheduled_arrival_ns is None:
            scheduled_arrival_ns = call_ns
        if scheduled_arrival_ns > call_ns:
            raise ValueError("runner called before the scheduled arrival")
        with self._lock:
            if request.request_id in self._pinned:
                raise RouteError("request is already pinned")
            self._pinned[request.request_id] = (
                request.route_epoch, request.route_id,
            )

        leases: dict[str, int] = {}
        touched: set[str] = set()
        boundaries: list[BoundaryActivation] = []
        primary: BaseException | None = None
        outcome: RouteOutcome | None = None
        deadline_ns = call_ns + int(timeout_s * 1e9)

        try:
            for stage in sorted(
                route.stages, key=lambda item: item.worker_name,
            ):
                while True:
                    seq_id = stage.slots.try_acquire(request.request_id)
                    if seq_id is not None:
                        leases[stage.worker_name] = seq_id
                        break
                    if time.monotonic_ns() >= deadline_ns:
                        raise TimeoutError(
                            f"sequence lease timed out on {stage.worker_name}"
                        )
                    time.sleep(0.001)
            lease_ready_ns = time.monotonic_ns()

            def propagate(
                inputs: Sequence[tuple[int, int, tuple[float, ...] | None]],
            ) -> tuple[BatchResult, ...]:
                stage_inputs = list(inputs)
                results: tuple[BatchResult, ...] = ()
                for stage_index, stage in enumerate(route.stages):
                    seq_id = leases[stage.worker_name]
                    upstream = (
                        "TOKEN_SOURCE" if stage_index == 0
                        else route.stages[stage_index - 1].worker_name
                    )
                    futures: list[Future[BatchResult]] = []
                    for position, token, hidden in stage_inputs:
                        touched.add(stage.worker_name)
                        futures.append(stage.batcher.submit(
                            BatchRow(
                                request.request_id,
                                request.route_epoch,
                                seq_id,
                                position,
                                token,
                                hidden,
                            ),
                            request.route_id,
                            upstream,
                            max(
                                0.001,
                                (deadline_ns - time.monotonic_ns()) / 1e9,
                            ),
                            request.batch_wait_us,
                            request.priority,
                        ))
                    results = tuple(
                        future.result(timeout=max(
                            0.001,
                            (deadline_ns - time.monotonic_ns()) / 1e9,
                        ))
                        for future in futures
                    )
                    if len(results) != len(stage_inputs):
                        raise ProtocolError(
                            f"{stage.worker_name} result count mismatch"
                        )
                    next_inputs = []
                    for source, result in zip(stage_inputs, results):
                        position, token, _hidden = source
                        expected = (
                            request.request_id,
                            request.route_epoch,
                            seq_id,
                            position,
                        )
                        actual = (
                            result.request_id,
                            result.route_epoch,
                            result.seq_id,
                            result.position,
                        )
                        if actual != expected:
                            raise ProtocolError(
                                f"{stage.worker_name} lineage mismatch"
                            )
                        if stage.terminal:
                            if result.token is None or result.hidden is not None:
                                raise ProtocolError(
                                    f"{stage.worker_name} did not return a token"
                                )
                            next_inputs.append((position, token, None))
                        else:
                            if result.hidden is None or result.token is not None:
                                raise ProtocolError(
                                    f"{stage.worker_name} did not return hidden state"
                                )
                            if not result.hidden or not all(
                                math.isfinite(value) for value in result.hidden
                            ):
                                raise ProtocolError(
                                    f"{stage.worker_name} returned non-finite hidden state"
                                )
                            if capture_boundaries:
                                boundaries.append(BoundaryActivation(
                                    stage.worker_name,
                                    stage.layer_end,
                                    position,
                                    result.hidden,
                                ))
                            next_inputs.append(
                                (position, token, result.hidden)
                            )
                    stage_inputs = next_inputs
                return results

            chunk = request.prefill_chunk or len(request.prompt_tokens)
            terminal_prefill: list[BatchResult] = []
            for start in range(0, len(request.prompt_tokens), chunk):
                stop = min(start + chunk, len(request.prompt_tokens))
                terminal_prefill.extend(propagate([
                    (position, request.prompt_tokens[position], None)
                    for position in range(start, stop)
                ]))
            if not terminal_prefill or terminal_prefill[-1].token is None:
                raise ProtocolError("terminal stage omitted prefill token")
            next_token = terminal_prefill[-1].token
            output_tokens = [next_token]
            first_token_ns = time.monotonic_ns()

            for output_index in range(1, request.output_steps):
                position = len(request.prompt_tokens) + output_index - 1
                decoded = propagate([(position, next_token, None)])
                if len(decoded) != 1 or decoded[0].token is None:
                    raise ProtocolError("terminal stage omitted decode token")
                next_token = decoded[0].token
                output_tokens.append(next_token)

            completed_ns = time.monotonic_ns()
            outcome = RouteOutcome(
                request_id=request.request_id,
                route_epoch=request.route_epoch,
                route_id=request.route_id,
                priority=request.priority,
                prompt_length=len(request.prompt_tokens),
                output_tokens=tuple(output_tokens),
                boundaries=tuple(boundaries),
                scheduled_arrival_ns=scheduled_arrival_ns,
                call_ns=call_ns,
                lease_ready_ns=lease_ready_ns,
                first_token_ns=first_token_ns,
                completed_ns=completed_ns,
                slo_us=request.slo_us,
            )
        except BaseException as exc:
            primary = exc

        cleanup_errors: list[BaseException] = []
        for stage in reversed(route.stages):
            seq_id = leases.get(stage.worker_name)
            if seq_id is None:
                continue
            if stage.worker_name in touched:
                try:
                    stage.client.remove(
                        seq_id, request.request_id, request.route_epoch,
                    )
                except BaseException as exc:
                    cleanup_errors.append(exc)
                    continue
            try:
                stage.slots.release(seq_id, request.request_id)
            except BaseException as exc:
                cleanup_errors.append(exc)

        if not cleanup_errors:
            with self._lock:
                pinned = self._pinned.get(request.request_id)
                expected = (request.route_epoch, request.route_id)
                if pinned != expected:
                    cleanup_errors.append(
                        RouteError("route pin changed during execution")
                    )
                else:
                    del self._pinned[request.request_id]

        if cleanup_errors:
            raise RouteCleanupError(primary, cleanup_errors)
        if primary is not None:
            raise primary
        if outcome is None:
            raise RouteError("route completed without an outcome")
        return outcome

    def run_group(
        self,
        requests: Sequence[RouteRequest],
        timeout_s: float,
        scheduled_arrival_ns: Sequence[int] | None = None,
        capture_boundaries: bool = False,
    ) -> tuple[RouteOutcome, ...]:
        if timeout_s <= 0 or not requests:
            raise ValueError("group and timeout must be positive")
        request_list = list(requests)
        route_id = request_list[0].route_id
        route = self._routes.get(route_id)
        if route is None:
            raise RouteError(f"unknown route {route_id}")
        if (
            any(request.route_id != route_id for request in request_list)
            or len({request.request_id for request in request_list}) != len(request_list)
            or len({len(request.prompt_tokens) for request in request_list}) != 1
            or len({request.output_steps for request in request_list}) != 1
            or len({request.batch_wait_us for request in request_list}) != 1
            or any(request.prefill_chunk not in (None, 1) for request in request_list)
        ):
            raise RouteError("route group must be homogeneous and uniquely owned")

        call_ns = time.monotonic_ns()
        arrivals = (
            tuple(call_ns for _request in request_list)
            if scheduled_arrival_ns is None
            else tuple(scheduled_arrival_ns)
        )
        if len(arrivals) != len(request_list) or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > call_ns
            for value in arrivals
        ):
            raise ValueError("group arrival timeline is invalid")

        with self._lock:
            if any(request.request_id in self._pinned for request in request_list):
                raise RouteError("group contains an already pinned request")
            for request in request_list:
                self._pinned[request.request_id] = (
                    request.route_epoch, request.route_id,
                )

        leases: dict[str, dict[int, int]] = {
            stage.worker_name: {} for stage in route.stages
        }
        touched: set[tuple[str, int]] = set()
        boundaries: dict[int, list[BoundaryActivation]] = {
            request.request_id: [] for request in request_list
        }
        outcomes: tuple[RouteOutcome, ...] | None = None
        primary: BaseException | None = None
        deadline_ns = call_ns + int(timeout_s * 1e9)

        try:
            for stage in sorted(route.stages, key=lambda item: item.worker_name):
                for request in request_list:
                    while True:
                        seq_id = stage.slots.try_acquire(request.request_id)
                        if seq_id is not None:
                            leases[stage.worker_name][request.request_id] = seq_id
                            break
                        if time.monotonic_ns() >= deadline_ns:
                            raise TimeoutError(
                                f"sequence lease timed out on {stage.worker_name}"
                            )
                        time.sleep(0.001)
            lease_ready_ns = time.monotonic_ns()

            def propagate(
                inputs: dict[int, tuple[int, int, tuple[float, ...] | None]],
            ) -> dict[int, BatchResult]:
                stage_inputs = dict(inputs)
                terminal_results: dict[int, BatchResult] = {}
                for stage_index, stage in enumerate(route.stages):
                    upstream = (
                        "TOKEN_SOURCE" if stage_index == 0
                        else route.stages[stage_index - 1].worker_name
                    )
                    futures: list[tuple[RouteRequest, int, int, Future[BatchResult]]] = []
                    for request in request_list:
                        position, token, hidden = stage_inputs[request.request_id]
                        seq_id = leases[stage.worker_name][request.request_id]
                        touched.add((stage.worker_name, request.request_id))
                        future = stage.batcher.submit(
                            BatchRow(
                                request.request_id,
                                request.route_epoch,
                                seq_id,
                                position,
                                token,
                                hidden,
                            ),
                            request.route_id,
                            upstream,
                            max(0.001, (deadline_ns - time.monotonic_ns()) / 1e9),
                            request.batch_wait_us,
                            request.priority,
                        )
                        futures.append((request, position, token, future))

                    next_inputs = {}
                    for request, position, token, future in futures:
                        result = future.result(timeout=max(
                            0.001, (deadline_ns - time.monotonic_ns()) / 1e9,
                        ))
                        expected = (
                            request.request_id,
                            request.route_epoch,
                            leases[stage.worker_name][request.request_id],
                            position,
                        )
                        actual = (
                            result.request_id,
                            result.route_epoch,
                            result.seq_id,
                            result.position,
                        )
                        if actual != expected:
                            raise ProtocolError(
                                f"{stage.worker_name} group lineage mismatch"
                            )
                        if stage.terminal:
                            if result.token is None or result.hidden is not None:
                                raise ProtocolError(
                                    f"{stage.worker_name} did not return a group token"
                                )
                            terminal_results[request.request_id] = result
                        else:
                            if result.hidden is None or result.token is not None:
                                raise ProtocolError(
                                    f"{stage.worker_name} did not return group hidden state"
                                )
                            if not result.hidden or not all(
                                math.isfinite(value) for value in result.hidden
                            ):
                                raise ProtocolError(
                                    f"{stage.worker_name} returned non-finite group state"
                                )
                            if capture_boundaries:
                                boundaries[request.request_id].append(BoundaryActivation(
                                    stage.worker_name,
                                    stage.layer_end,
                                    position,
                                    result.hidden,
                                ))
                            next_inputs[request.request_id] = (
                                position, token, result.hidden,
                            )
                    if not stage.terminal:
                        stage_inputs = next_inputs
                if len(terminal_results) != len(request_list):
                    raise ProtocolError("terminal stage omitted a group result")
                return terminal_results

            terminal_prefill: dict[int, BatchResult] = {}
            prompt_length = len(request_list[0].prompt_tokens)
            for position in range(prompt_length):
                terminal_prefill = propagate({
                    request.request_id: (
                        position, request.prompt_tokens[position], None,
                    )
                    for request in request_list
                })
            next_tokens = {
                request.request_id: terminal_prefill[request.request_id].token
                for request in request_list
            }
            if any(token is None for token in next_tokens.values()):
                raise ProtocolError("terminal stage omitted a prefill group token")
            output_tokens = {
                request.request_id: [next_tokens[request.request_id]]
                for request in request_list
            }
            first_token_ns = time.monotonic_ns()

            for output_index in range(1, request_list[0].output_steps):
                position = prompt_length + output_index - 1
                decoded = propagate({
                    request.request_id: (
                        position, int(next_tokens[request.request_id]), None,
                    )
                    for request in request_list
                })
                for request in request_list:
                    token = decoded[request.request_id].token
                    if token is None:
                        raise ProtocolError("terminal stage omitted a decode group token")
                    next_tokens[request.request_id] = token
                    output_tokens[request.request_id].append(token)

            completed_ns = time.monotonic_ns()
            outcomes = tuple(
                RouteOutcome(
                    request_id=request.request_id,
                    route_epoch=request.route_epoch,
                    route_id=request.route_id,
                    priority=request.priority,
                    prompt_length=len(request.prompt_tokens),
                    output_tokens=tuple(output_tokens[request.request_id]),
                    boundaries=tuple(boundaries[request.request_id]),
                    scheduled_arrival_ns=arrival,
                    call_ns=call_ns,
                    lease_ready_ns=lease_ready_ns,
                    first_token_ns=first_token_ns,
                    completed_ns=completed_ns,
                    slo_us=request.slo_us,
                )
                for request, arrival in zip(request_list, arrivals)
            )
        except BaseException as exc:
            primary = exc

        cleanup_errors: list[BaseException] = []
        for stage in reversed(route.stages):
            for request in reversed(request_list):
                seq_id = leases[stage.worker_name].get(request.request_id)
                if seq_id is None:
                    continue
                if (stage.worker_name, request.request_id) in touched:
                    try:
                        stage.client.remove(
                            seq_id, request.request_id, request.route_epoch,
                        )
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                        continue
                try:
                    stage.slots.release(seq_id, request.request_id)
                except BaseException as exc:
                    cleanup_errors.append(exc)

        if not cleanup_errors:
            with self._lock:
                changed = [
                    request.request_id for request in request_list
                    if self._pinned.get(request.request_id) != (
                        request.route_epoch, request.route_id,
                    )
                ]
                if changed:
                    cleanup_errors.append(
                        RouteError("route group pin changed during execution")
                    )
                else:
                    for request in request_list:
                        del self._pinned[request.request_id]

        if cleanup_errors:
            raise RouteCleanupError(primary, cleanup_errors)
        if primary is not None:
            raise primary
        if outcomes is None:
            raise RouteError("route group completed without outcomes")
        return outcomes
