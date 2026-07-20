#!/usr/bin/env python3
"""Host-side mixed-workload dispatch coordinator for S15.

Wraps one PriorityBatchRuntime (callers never choose a batch), a
ReadyRouteRegistry (readiness, epoch, credit, and thermal fencing), and an
Executor (typed launch boundary). It routes each request to exactly one of four
isolated lanes -- selected-A6000 BGE, OP15 Gemma head, OP12 Gemma head, or
server fallback -- and drives every request to exactly one terminal state.

No device I/O, no ADB, no latency or energy claim. Measured durations are used
only for the policy's latest-start feasibility gate.
"""

from __future__ import annotations

import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_S14 = Path(__file__).resolve().parent.parent / "s14_mixed_streaming_scheduler"
if str(_S14) not in sys.path:
    sys.path.insert(0, str(_S14))

from power_frontier_policy import BatchDecision, WorkItem  # noqa: E402
from priority_batch_runtime import (  # noqa: E402
    BatchRuntimeError,
    Launch,
    PriorityBatchRuntime,
    RouteConfig,
)

from executor_contract import (  # noqa: E402
    BOUNDARY_SCHEMA,
    ExecutionRequest,
    Executor,
    ExecutorError,
)
from route_registry import ReadyRouteRegistry, RouteRegistryError  # noqa: E402


TERMINAL_STATES = (
    "completed_phone",
    "completed_server",
    "tardy_result",
    "fallback_required",
    "rejected_backpressure",
    "timed_out",
)
DEVICE_CLASSES = ("phone", "server")
SERVER_FALLBACK_LANE = "server_fallback"


class DispatchError(ValueError):
    pass


def _int(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DispatchError(f"{name} must be an integer >= {minimum}")
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise DispatchError(f"{name} must be a non-empty string")
    return value


def _sha256(name: str, value: object) -> str:
    text = _text(name, value)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise DispatchError(f"{name} must be a lowercase sha256 digest")
    return text


@dataclass(frozen=True)
class LaneBinding:
    lane: str
    route_id: str
    device_class: str
    timeout_us: int
    cohort_sha256: str
    input_manifest_sha256: str

    def validate(self) -> None:
        _text("lane", self.lane)
        _text("route_id", self.route_id)
        if self.lane == SERVER_FALLBACK_LANE:
            raise DispatchError("server fallback is a reserved sink lane")
        if self.device_class not in DEVICE_CLASSES:
            raise DispatchError(f"unknown device class {self.device_class!r}")
        _int("timeout_us", self.timeout_us, 1)
        _sha256("cohort_sha256", self.cohort_sha256)
        _sha256("input_manifest_sha256", self.input_manifest_sha256)


@dataclass(frozen=True)
class AdmitResult:
    request_id: str
    disposition: str  # "queued" | "server_fallback" | "rejected_backpressure"
    route_id: str | None
    reason: str
    terminal: str | None


class MixedDispatchCoordinator:
    def __init__(
        self,
        registry: ReadyRouteRegistry,
        executor: Executor,
        lanes: Sequence[LaneBinding],
        *,
        queue_capacity: int,
    ) -> None:
        if type(registry) is not ReadyRouteRegistry:
            raise DispatchError("registry must be a ReadyRouteRegistry")
        if not isinstance(executor, Executor):
            raise DispatchError("executor must implement the Executor interface")
        _int("queue_capacity", queue_capacity, 1)
        if not lanes:
            raise DispatchError("at least one lane is required")
        self._registry = registry
        self._executor = executor
        self._queue_capacity = queue_capacity
        self._lanes: dict[str, LaneBinding] = {}
        self._signature_index: dict[tuple[str, str, str], str] = {}
        configs: list[RouteConfig] = []
        for binding in lanes:
            if type(binding) is not LaneBinding:
                raise DispatchError("lanes must contain LaneBinding values")
            binding.validate()
            if binding.route_id in self._lanes:
                raise DispatchError(f"duplicate lane route {binding.route_id!r}")
            snapshot = registry.get(binding.route_id)
            if snapshot is None:
                raise DispatchError(f"lane route {binding.route_id!r} is not in the registry")
            config = snapshot.config
            signature = (config.service_class, config.model_id, config.island_id)
            if signature in self._signature_index:
                raise DispatchError(f"lane signature {signature} is aliased")
            self._signature_index[signature] = binding.route_id
            self._lanes[binding.route_id] = binding
            configs.append(config)
        self._runtime = PriorityBatchRuntime(configs)
        self._terminal: dict[str, str] = {}
        self._seen: set[str] = set()
        self._pending: dict[str, dict[str, WorkItem]] = {
            route_id: {} for route_id in self._lanes
        }
        self._fallback_queue: list[str] = []

    # -- admission ---------------------------------------------------------

    def admit(self, item: WorkItem, now_us: int) -> AdmitResult:
        if type(item) is not WorkItem:
            raise DispatchError("item must be a WorkItem")
        _int("now_us", now_us)
        try:
            item.validate(now_us)
        except Exception as exc:  # PolicyError
            raise DispatchError(str(exc)) from exc
        if item.request_id in self._seen:
            raise DispatchError(f"request {item.request_id!r} was already admitted")
        self._seen.add(item.request_id)

        route_id = self._signature_index.get(
            (item.service_class, item.model_id, item.island_id)
        )
        if route_id is None:
            return self._server_fallback(item, now_us, "no_matching_route")

        binding = self._lanes[route_id]
        snapshot = self._registry.get(route_id)
        if snapshot is None or not snapshot.dispatchable():
            if binding.device_class == "phone":
                return self._server_fallback(item, now_us, "route_not_ready")
            return self._reject(item, "server_route_unavailable")

        if binding.device_class == "phone" and not self._phone_feasible(snapshot, item, now_us):
            return self._server_fallback(item, now_us, "latest_start_infeasible")

        if len(self._pending[route_id]) >= self._queue_capacity:
            return self._reject(item, "queue_full")

        try:
            self._runtime.enqueue(route_id, item, now_us)
        except BatchRuntimeError as exc:
            raise DispatchError(str(exc)) from exc
        self._pending[route_id][item.request_id] = item
        return AdmitResult(item.request_id, "queued", route_id, "enqueued", None)

    def _phone_feasible(self, snapshot, item: WorkItem, now_us: int) -> bool:
        min_duration = min(point.duration_us for point in snapshot.config.points)
        return now_us + min_duration <= item.deadline_us

    def _server_fallback(self, item: WorkItem, now_us: int, reason: str) -> AdmitResult:
        self._fallback_queue.append(item.request_id)
        terminal = "tardy_result" if now_us > item.deadline_us else "completed_server"
        self._set_terminal(item.request_id, terminal)
        return AdmitResult(item.request_id, "server_fallback", SERVER_FALLBACK_LANE, reason, terminal)

    def _reject(self, item: WorkItem, reason: str) -> AdmitResult:
        self._set_terminal(item.request_id, "rejected_backpressure")
        return AdmitResult(item.request_id, "rejected_backpressure", None, reason, "rejected_backpressure")

    # -- dispatch ----------------------------------------------------------

    def dispatch(self, route_id: str, now_us: int) -> Launch | BatchDecision:
        _text("route_id", route_id)
        _int("now_us", now_us)
        if route_id not in self._lanes:
            raise DispatchError(f"unknown lane route {route_id!r}")
        decision = self._runtime.decide(route_id, now_us)
        if isinstance(decision, BatchDecision):
            if decision.action == "NO_FEASIBLE":
                raise DispatchError(
                    "runtime reported an infeasible batch after feasible admission"
                )
            return decision

        launch = decision
        binding = self._lanes[route_id]
        snapshot = self._registry.get(route_id)
        try:
            lease = self._registry.acquire_lease(route_id, now_us)
        except RouteRegistryError:
            self._fail_launch(route_id, launch, now_us, "completed_server")
            return launch

        request = ExecutionRequest(
            launch_id=launch.launch_id,
            route_id=route_id,
            profile_id=lease.profile_id,
            device_id=lease.device_id,
            route_epoch=lease.route_epoch,
            residency_epoch=lease.residency_epoch,
            lease_epoch=lease.lease_epoch,
            device_boot_epoch=lease.device_boot_epoch,
            registry_generation=lease.generation,
            compatibility_key=launch.compatibility_key,
            request_ids=launch.request_ids,
            cohort_sha256=binding.cohort_sha256,
            input_manifest_sha256=binding.input_manifest_sha256,
            timeout_us=binding.timeout_us,
            expected_boundary_schema=BOUNDARY_SCHEMA,
        )
        try:
            result = self._executor.launch(request, now_us)
            result.validate(request)
        except ExecutorError:
            self._registry.release_lease(lease)
            self._fail_launch(route_id, launch, now_us, "fallback_required")
            return launch

        if not self._registry.validate_lease(lease):
            self._registry.release_lease(lease)
            self._fail_launch(route_id, launch, result.finish_us, "fallback_required")
            return launch
        self._registry.release_lease(lease)

        if result.outcome == "timed_out":
            self._fail_launch(route_id, launch, result.finish_us, "timed_out")
            return launch
        if result.outcome == "error":
            self._fail_launch(route_id, launch, result.finish_us, "fallback_required")
            return launch

        try:
            completions = self._runtime.complete(
                route_id, launch.launch_id, launch.route_epoch,
                result.finish_us, result.certificates,
            )
        except BatchRuntimeError:
            self._fail_launch(route_id, launch, result.finish_us, "fallback_required")
            return launch
        for completion in completions:
            terminal = self._map_status(completion.status, binding.device_class)
            self._set_terminal(completion.request_id, terminal)
            self._pending[route_id].pop(completion.request_id, None)
        return launch

    def _fail_launch(self, route_id: str, launch: Launch, finish_us: int, terminal: str) -> None:
        self._runtime.fail_launch(route_id, launch.launch_id, launch.route_epoch, finish_us)
        for request_id in launch.request_ids:
            self._set_terminal(request_id, terminal)
            self._pending[route_id].pop(request_id, None)
            if terminal in ("completed_server", "fallback_required"):
                self._fallback_queue.append(request_id)

    @staticmethod
    def _map_status(status: str, device_class: str) -> str:
        if status == "completed":
            return "completed_phone" if device_class == "phone" else "completed_server"
        if status == "tardy_result":
            return "tardy_result"
        if status == "fallback_required":
            return "fallback_required"
        raise DispatchError(f"unmapped runtime status {status!r}")

    # -- ledger ------------------------------------------------------------

    def _set_terminal(self, request_id: str, terminal: str) -> None:
        _text("request_id", request_id)
        if terminal not in TERMINAL_STATES:
            raise DispatchError(f"unknown terminal state {terminal!r}")
        if request_id in self._terminal:
            raise DispatchError(f"request {request_id!r} already has a terminal state")
        self._terminal[request_id] = terminal

    def terminal_of(self, request_id: str) -> str | None:
        return self._terminal.get(request_id)

    def terminals(self) -> dict[str, str]:
        return dict(self._terminal)

    def fallback_queue(self) -> tuple[str, ...]:
        return tuple(self._fallback_queue)

    def pending_count(self, route_id: str) -> int:
        _text("route_id", route_id)
        if route_id not in self._pending:
            raise DispatchError(f"unknown lane route {route_id!r}")
        return len(self._pending[route_id])

    def assert_conservation(self, expected_request_ids: Sequence[str]) -> None:
        expected = list(expected_request_ids)
        if len(expected) != len(set(expected)):
            raise DispatchError("expected request ids are not unique")
        expected_set = set(expected)
        terminal_set = set(self._terminal)
        if terminal_set != expected_set:
            missing = sorted(expected_set - terminal_set)
            extra = sorted(terminal_set - expected_set)
            raise DispatchError(f"conservation failed: missing={missing} extra={extra}")
        for request_id, terminal in self._terminal.items():
            if terminal not in TERMINAL_STATES:
                raise DispatchError(f"request {request_id!r} has an invalid terminal state")
