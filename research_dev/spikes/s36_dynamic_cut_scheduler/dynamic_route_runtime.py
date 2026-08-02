#!/usr/bin/env python3
"""Exact dynamic-cut routes over real StageNet V3 workers."""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
S23 = HERE.parent / "s23_dense_trace_runtime"
for dependency in (S23, S22):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from runtime_support import SequenceSlotPool  # noqa: E402
from stage_v3_client import BatchResult, BatchRow, Hello, ProtocolError  # noqa: E402

from cut_batcher import CutBatcher, LockedStageClient, RequestCutTable  # noqa: E402


class DynamicRouteError(RuntimeError):
    pass


@dataclass(frozen=True)
class DynamicStage:
    name: str
    client: LockedStageClient
    hello: Hello
    slots: SequenceSlotPool
    batcher: CutBatcher
    terminal: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name is required")
        if self.slots.capacity != self.hello.max_streams:
            raise ValueError("stage slot capacity differs from worker")


@dataclass(frozen=True)
class DynamicRoute:
    route_id: str
    head: DynamicStage
    tail: DynamicStage
    cut: int

    def __post_init__(self) -> None:
        if not self.route_id or type(self.cut) is not int or self.cut <= 0:
            raise ValueError("invalid dynamic route")
        if self.head.terminal or not self.tail.terminal:
            raise ValueError("dynamic route requires a head and terminal tail")
        if self.cut not in self.head.batcher.ranges:
            raise ValueError("route cut is not resident on its head")
        if self.cut not in self.tail.batcher.ranges:
            raise ValueError("route cut is not resident on its tail")
        if self.head.batcher.ranges[self.cut] != (0, self.cut):
            raise ValueError("head range differs from route cut")
        if self.tail.batcher.ranges[self.cut][0] != self.cut:
            raise ValueError("tail range differs from route cut")
        if self.tail.batcher.ranges[self.cut][1] != self.tail.hello.n_layer:
            raise ValueError("route does not reach the final layer")


@dataclass(frozen=True)
class DynamicRequest:
    request_id: int
    route_epoch: int
    route_id: str
    prompt_tokens: tuple[int, ...]
    output_steps: int
    priority: int
    slo_us: int
    batch_wait_us: int
    prefill_quantum: int = 64
    stop_tokens: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.request_id) is not int
            or self.request_id <= 0
            or type(self.route_epoch) is not int
            or self.route_epoch <= 0
            or not self.route_id
            or not self.prompt_tokens
            or any(type(token) is not int or token < 0 for token in self.prompt_tokens)
            or type(self.output_steps) is not int
            or self.output_steps <= 0
            or type(self.priority) is not int
            or self.priority < 0
            or type(self.slo_us) is not int
            or self.slo_us <= 0
            or type(self.batch_wait_us) is not int
            or self.batch_wait_us < 0
            or type(self.prefill_quantum) is not int
            or not 1 <= self.prefill_quantum <= 64
            or type(self.stop_tokens) is not tuple
            or any(type(token) is not int or token < 0 for token in self.stop_tokens)
            or len(set(self.stop_tokens)) != len(self.stop_tokens)
        ):
            raise ValueError("invalid dynamic request")


@dataclass(frozen=True)
class DynamicOutcome:
    request_id: int
    route_epoch: int
    route_id: str
    device: str
    cut: int
    priority: int
    prompt_length: int
    output_tokens: tuple[int, ...]
    scheduled_arrival_ns: int
    call_ns: int
    lease_ready_ns: int
    first_token_ns: int
    completed_ns: int
    slo_us: int
    finish_reason: str = "length"

    @property
    def ttft_us(self) -> int:
        return (self.first_token_ns - self.scheduled_arrival_ns) // 1000

    @property
    def latency_us(self) -> int:
        return (self.completed_ns - self.scheduled_arrival_ns) // 1000

    @property
    def slo_met(self) -> bool:
        return self.latency_us <= self.slo_us


class DynamicRouteRunner:
    def __init__(self, routes: Sequence[DynamicRoute]) -> None:
        by_id = {route.route_id: route for route in routes}
        if not routes or len(by_id) != len(routes):
            raise ValueError("dynamic routes must be nonempty and unique")
        self._routes = by_id
        self._pins: dict[int, tuple[int, str, int]] = {}
        self._cut_table = RequestCutTable()
        self._lock = threading.Lock()

    @property
    def routes(self) -> Mapping[str, DynamicRoute]:
        return dict(self._routes)

    def pins(self) -> dict[int, tuple[int, str, int]]:
        with self._lock:
            return dict(self._pins)

    @staticmethod
    def _wait_result(future: object, deadline_ns: int) -> BatchResult:
        remaining_s = (deadline_ns - time.monotonic_ns()) / 1e9
        if remaining_s <= 0:
            raise TimeoutError("dynamic route request timed out")
        result = future.result(timeout=remaining_s)
        if not isinstance(result, BatchResult):
            raise ProtocolError("stage returned an invalid result type")
        return result

    @staticmethod
    def _validate_lineage(source: BatchRow, result: BatchResult) -> None:
        expected = (
            source.request_id, source.route_epoch, source.seq_id, source.position,
        )
        actual = (
            result.request_id, result.route_epoch, result.seq_id, result.position,
        )
        if actual != expected:
            raise ProtocolError("dynamic route result lineage mismatch")

    def run(
        self,
        request: DynamicRequest,
        timeout_s: float,
        scheduled_arrival_ns: int | None = None,
    ) -> DynamicOutcome:
        if (
            type(timeout_s) not in (int, float)
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout must be positive and finite")
        route = self._routes.get(request.route_id)
        if route is None:
            raise DynamicRouteError("unknown dynamic route")
        call_ns = time.monotonic_ns()
        if scheduled_arrival_ns is None:
            scheduled_arrival_ns = call_ns
        if (
            type(scheduled_arrival_ns) is not int
            or scheduled_arrival_ns < 0
            or scheduled_arrival_ns > call_ns
        ):
            raise ValueError("scheduled arrival is invalid")
        deadline_ns = call_ns + int(timeout_s * 1e9)

        with self._lock:
            if request.request_id in self._pins:
                raise DynamicRouteError("request is already pinned")
            self._pins[request.request_id] = (
                request.route_epoch, request.route_id, route.cut,
            )
        self._cut_table.pin(request.request_id, request.route_epoch, route.cut)

        leases: dict[str, int] = {}
        touched: set[str] = set()
        primary: BaseException | None = None
        outcome: DynamicOutcome | None = None
        try:
            for stage in sorted((route.head, route.tail), key=lambda item: item.name):
                while True:
                    seq_id = stage.slots.try_acquire(request.request_id)
                    if seq_id is not None:
                        leases[stage.name] = seq_id
                        break
                    if time.monotonic_ns() >= deadline_ns:
                        raise TimeoutError(f"sequence lease timed out on {stage.name}")
                    time.sleep(0.001)
            lease_ready_ns = time.monotonic_ns()

            def propagate(
                inputs: Sequence[tuple[int, int]], phase: str,
            ) -> tuple[BatchResult, ...]:
                self._cut_table.require(
                    request.request_id, request.route_epoch, route.cut,
                )
                head_rows = [
                    BatchRow(
                        request.request_id,
                        request.route_epoch,
                        leases[route.head.name],
                        position,
                        token,
                    )
                    for position, token in inputs
                ]
                touched.add(route.head.name)
                head_futures = [
                    route.head.batcher.submit(
                        row,
                        route.cut,
                        phase,
                        max(0.001, (deadline_ns - time.monotonic_ns()) / 1e9),
                        request.batch_wait_us,
                        request.priority,
                    )
                    for row in head_rows
                ]
                head_results = tuple(
                    self._wait_result(future, deadline_ns) for future in head_futures
                )
                tail_rows = []
                for source, result in zip(head_rows, head_results):
                    self._validate_lineage(source, result)
                    if (
                        result.hidden is None
                        or result.token is not None
                        or not result.hidden
                        or not all(math.isfinite(value) for value in result.hidden)
                    ):
                        raise ProtocolError("head returned an invalid activation")
                    tail_rows.append(BatchRow(
                        request.request_id,
                        request.route_epoch,
                        leases[route.tail.name],
                        source.position,
                        source.token,
                        result.hidden,
                    ))

                touched.add(route.tail.name)
                tail_futures = [
                    route.tail.batcher.submit(
                        row,
                        route.cut,
                        phase,
                        max(0.001, (deadline_ns - time.monotonic_ns()) / 1e9),
                        request.batch_wait_us,
                        request.priority,
                    )
                    for row in tail_rows
                ]
                tail_results = tuple(
                    self._wait_result(future, deadline_ns) for future in tail_futures
                )
                for source, result in zip(tail_rows, tail_results):
                    self._validate_lineage(source, result)
                    if result.hidden is not None or type(result.token) is not int:
                        raise ProtocolError("tail returned an invalid token")
                return tail_results

            prompt_results: list[BatchResult] = []
            for begin in range(0, len(request.prompt_tokens), request.prefill_quantum):
                end = min(begin + request.prefill_quantum, len(request.prompt_tokens))
                prompt_results.extend(propagate(
                    tuple(
                        (position, request.prompt_tokens[position])
                        for position in range(begin, end)
                    ),
                    "prefill",
                ))
            if len(prompt_results) != len(request.prompt_tokens):
                raise ProtocolError("prompt result count mismatch")
            next_token = prompt_results[-1].token
            if type(next_token) is not int:
                raise ProtocolError("terminal prompt token is missing")
            output_tokens = [next_token]
            first_token_ns = time.monotonic_ns()
            finish_reason = "stop" if next_token in request.stop_tokens else "length"

            for output_index in range(1, request.output_steps):
                if finish_reason == "stop":
                    break
                position = len(request.prompt_tokens) + output_index - 1
                decoded = propagate(((position, next_token),), "decode")
                if len(decoded) != 1 or type(decoded[0].token) is not int:
                    raise ProtocolError("terminal decode token is missing")
                next_token = decoded[0].token
                output_tokens.append(next_token)
                if next_token in request.stop_tokens:
                    finish_reason = "stop"

            completed_ns = time.monotonic_ns()
            outcome = DynamicOutcome(
                request_id=request.request_id,
                route_epoch=request.route_epoch,
                route_id=request.route_id,
                device=route.head.name,
                cut=route.cut,
                priority=request.priority,
                prompt_length=len(request.prompt_tokens),
                output_tokens=tuple(output_tokens),
                scheduled_arrival_ns=scheduled_arrival_ns,
                call_ns=call_ns,
                lease_ready_ns=lease_ready_ns,
                first_token_ns=first_token_ns,
                completed_ns=completed_ns,
                slo_us=request.slo_us,
                finish_reason=finish_reason,
            )
        except BaseException as exc:
            primary = exc

        cleanup_errors: list[BaseException] = []
        for stage in (route.tail, route.head):
            seq_id = leases.get(stage.name)
            if seq_id is None:
                continue
            if stage.name in touched:
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
            try:
                self._cut_table.remove(request.request_id, request.route_epoch)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if not cleanup_errors:
            with self._lock:
                expected = (request.route_epoch, request.route_id, route.cut)
                if self._pins.get(request.request_id) != expected:
                    cleanup_errors.append(DynamicRouteError("request pin changed"))
                else:
                    del self._pins[request.request_id]

        if cleanup_errors:
            error = DynamicRouteError(
                f"dynamic route cleanup failed in {len(cleanup_errors)} operation(s)"
            )
            if primary is not None:
                error.__cause__ = primary
            raise error
        if primary is not None:
            raise primary
        if outcome is None:
            raise DynamicRouteError("dynamic route completed without outcome")
        return outcome
