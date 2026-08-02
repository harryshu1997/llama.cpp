#!/usr/bin/env python3
"""Cut-homogeneous continuous batching for one StageNet V3 worker."""

from __future__ import annotations

import threading
import time
import math
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Mapping, Sequence


MAX_PRIORITY = (1 << 31) - 1


class CutBatcherError(RuntimeError):
    pass


class LockedStageClient:
    """Serialize all operations on one StageNet socket."""

    def __init__(self, client: object):
        self._client = client
        self._lock = threading.Lock()

    def _call(self, name: str, *args: object) -> object:
        with self._lock:
            return getattr(self._client, name)(*args)

    def hello(self) -> object:
        return self._call("hello")

    def range_batch(
        self, rows: Sequence[object], layer_start: int, layer_end: int,
    ) -> tuple[object, ...]:
        return tuple(self._call("range_batch", rows, layer_start, layer_end))

    def remove(self, seq_id: int, request_id: int, route_epoch: int) -> object:
        return self._call("remove", seq_id, request_id, route_epoch)

    def status(self) -> object:
        return self._call("status")

    def drain(self) -> object:
        return self._call("drain")

    def stop(self) -> None:
        self._call("stop")

    def detach(self) -> None:
        self._call("detach")

    def close(self) -> None:
        self._call("close")


@dataclass
class PendingRow:
    row: object
    future: Future[object]
    cut: int
    phase: str
    priority: int
    enqueued_ns: int
    latest_dispatch_ns: int
    order: int


class RequestCutTable:
    """Pin one cut to a live request epoch until explicit removal."""

    def __init__(self) -> None:
        self._cuts: dict[tuple[int, int], int] = {}
        self._lock = threading.Lock()

    def pin(self, request_id: int, route_epoch: int, cut: int) -> None:
        key = self._key(request_id, route_epoch)
        self._validate_cut(cut)
        with self._lock:
            current = self._cuts.get(key)
            if current is not None and current != cut:
                raise CutBatcherError("live request cannot change cut")
            self._cuts[key] = cut

    def require(self, request_id: int, route_epoch: int, cut: int) -> None:
        key = self._key(request_id, route_epoch)
        self._validate_cut(cut)
        with self._lock:
            if self._cuts.get(key) != cut:
                raise CutBatcherError("request cut is not pinned")

    def remove(self, request_id: int, route_epoch: int) -> None:
        key = self._key(request_id, route_epoch)
        with self._lock:
            if key not in self._cuts:
                raise CutBatcherError("request cut removal is not live")
            del self._cuts[key]

    def snapshot(self) -> dict[tuple[int, int], int]:
        with self._lock:
            return dict(self._cuts)

    @staticmethod
    def _key(request_id: int, route_epoch: int) -> tuple[int, int]:
        if (
            type(request_id) is not int
            or type(route_epoch) is not int
            or request_id < 0
            or route_epoch < 0
        ):
            raise CutBatcherError("invalid request lineage")
        return request_id, route_epoch

    @staticmethod
    def _validate_cut(cut: int) -> None:
        if type(cut) is not int or cut <= 0:
            raise CutBatcherError("invalid request cut")


class CutBatcher:
    """Batch one physical worker without mixing cuts or priority bands."""

    def __init__(
        self,
        name: str,
        client: LockedStageClient,
        ranges: Mapping[int, tuple[int, int]],
        knees: Mapping[int, int],
        max_rows: int,
        gather_us: int,
        queue_depth: int,
    ) -> None:
        if not name or type(name) is not str:
            raise ValueError("batcher name is required")
        if (
            type(max_rows) is not int
            or type(gather_us) is not int
            or type(queue_depth) is not int
            or max_rows <= 0
            or gather_us < 0
            or queue_depth <= 0
        ):
            raise ValueError("invalid batcher bounds")
        normalized_ranges: dict[int, tuple[int, int]] = {}
        for cut, active_range in ranges.items():
            if (
                type(cut) is not int
                or cut <= 0
                or type(active_range) is not tuple
                or len(active_range) != 2
                or any(type(value) is not int for value in active_range)
                or active_range[0] < 0
                or active_range[0] >= active_range[1]
                or cut not in active_range
            ):
                raise ValueError("invalid active range")
            normalized_ranges[cut] = active_range
        if not normalized_ranges or set(normalized_ranges) != set(knees):
            raise ValueError("every cut requires one range and knee")
        if any(type(value) is not int or not 1 <= value <= max_rows for value in knees.values()):
            raise ValueError("invalid cut knee")

        self.name = name
        self.client = client
        self.ranges = normalized_ranges
        self.knees = dict(knees)
        self.max_rows = max_rows
        self.gather_us = gather_us
        self.queue_depth = queue_depth
        self.events: list[dict[str, object]] = []

        self._condition = threading.Condition()
        self._pending: list[PendingRow] = []
        self._error: BaseException | None = None
        self._stopping = False
        self._stopped = False
        self._next_order = 0
        self._thread = threading.Thread(target=self._run, name=f"cut-batch-{name}")
        self._thread.start()

    @staticmethod
    def _priority_band(priority: int) -> int:
        return 0 if priority == 0 else 1

    def submit(
        self,
        row: object,
        cut: int,
        phase: str,
        timeout_s: float,
        batch_wait_us: int | None = None,
        priority: int = 0,
    ) -> Future[object]:
        if cut not in self.ranges:
            raise ValueError("cut is not configured")
        if phase not in ("prefill", "decode"):
            raise ValueError("phase must be prefill or decode")
        if (
            type(timeout_s) not in (int, float)
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout must be positive")
        if batch_wait_us is not None and (
            type(batch_wait_us) is not int or batch_wait_us < 0
        ):
            raise ValueError("batch wait budget cannot be negative")
        if type(priority) is not int or not 0 <= priority <= MAX_PRIORITY:
            raise ValueError("priority is out of range")

        future: Future[object] = Future()
        deadline_ns = time.monotonic_ns() + int(timeout_s * 1e9)
        with self._condition:
            while len(self._pending) >= self.queue_depth:
                self._raise_if_unavailable()
                remaining_s = (deadline_ns - time.monotonic_ns()) / 1e9
                if remaining_s <= 0:
                    raise TimeoutError(f"{self.name} admission queue is full")
                self._condition.wait(remaining_s)
            self._raise_if_unavailable()
            enqueued_ns = time.monotonic_ns()
            wait_us = self.gather_us
            if batch_wait_us is not None:
                wait_us = min(wait_us, batch_wait_us)
            self._pending.append(PendingRow(
                row=row,
                future=future,
                cut=cut,
                phase=phase,
                priority=priority,
                enqueued_ns=enqueued_ns,
                latest_dispatch_ns=enqueued_ns + wait_us * 1000,
                order=self._next_order,
            ))
            self._next_order += 1
            self._condition.notify_all()
        return future

    def _raise_if_unavailable(self) -> None:
        if self._error is not None:
            raise CutBatcherError(f"{self.name} batcher failed") from self._error
        if self._stopping:
            raise CutBatcherError(f"{self.name} batcher is stopping")

    def pending_by_cut(self) -> dict[int, int]:
        with self._condition:
            return {
                cut: sum(item.cut == cut for item in self._pending)
                for cut in sorted(self.ranges)
            }

    def _groups(self) -> dict[tuple[int, int], list[PendingRow]]:
        groups: dict[tuple[int, int], list[PendingRow]] = {}
        for item in self._pending:
            groups.setdefault(
                (item.cut, self._priority_band(item.priority)), [],
            ).append(item)
        for items in groups.values():
            items.sort(key=lambda item: item.order)
        return groups

    def _select(
        self, now_ns: int,
    ) -> tuple[list[PendingRow], str, int | None]:
        groups = self._groups()
        if not groups:
            return [], "", None

        candidates: list[tuple[int, int, int, int, list[PendingRow], str]] = []
        next_deadline_ns: int | None = None
        for (cut, band), items in groups.items():
            earliest_ns = min(item.latest_dispatch_ns for item in items)
            next_deadline_ns = (
                earliest_ns
                if next_deadline_ns is None
                else min(next_deadline_ns, earliest_ns)
            )
            reason = ""
            if self._stopping:
                reason = "DRAIN"
            elif len(items) >= self.knees[cut]:
                reason = "BATCH_KNEE"
            elif earliest_ns <= now_ns:
                reason = "LATEST_SAFE_START"
            if reason:
                candidates.append((
                    band,
                    earliest_ns,
                    -min(len(items), self.max_rows),
                    cut,
                    items,
                    reason,
                ))
        if not candidates:
            return [], "", next_deadline_ns
        candidates.sort(key=lambda item: item[:4])
        chosen = candidates[0]
        return chosen[4][:self.max_rows], chosen[5], next_deadline_ns

    def _fail_all(self, error: BaseException) -> None:
        for item in self._pending:
            if not item.future.done():
                item.future.set_exception(error)
        self._pending.clear()
        self._condition.notify_all()

    def _run(self) -> None:
        active: list[PendingRow] = []
        try:
            while True:
                with self._condition:
                    while True:
                        now_ns = time.monotonic_ns()
                        active, reason, next_deadline_ns = self._select(now_ns)
                        if active:
                            selected = {item.order for item in active}
                            self._pending = [
                                item for item in self._pending
                                if item.order not in selected
                            ]
                            self._condition.notify_all()
                            break
                        if self._stopping and not self._pending:
                            self._stopped = True
                            self._condition.notify_all()
                            return
                        timeout_s = None
                        if next_deadline_ns is not None:
                            timeout_s = max(
                                0.0, (next_deadline_ns - now_ns) / 1e9,
                            )
                        self._condition.wait(timeout_s)

                compute_start_ns = time.monotonic_ns()
                active_range = self.ranges[active[0].cut]
                results = self.client.range_batch(
                    [item.row for item in active], *active_range,
                )
                compute_end_ns = time.monotonic_ns()
                if len(results) != len(active):
                    raise CutBatcherError(f"{self.name} result count mismatch")
                phases = [item.phase for item in active]
                self.events.append({
                    "worker": self.name,
                    "cut": active[0].cut,
                    "active_range": list(active_range),
                    "priority_band": self._priority_band(active[0].priority),
                    "batch_size": len(active),
                    "phases": phases,
                    "mixed_phase": len(set(phases)) > 1,
                    "request_ids": [getattr(item.row, "request_id") for item in active],
                    "route_epochs": [getattr(item.row, "route_epoch") for item in active],
                    "positions": [getattr(item.row, "position") for item in active],
                    "priorities": [item.priority for item in active],
                    "release_reason": reason,
                    "compute_us": (compute_end_ns - compute_start_ns) // 1000,
                    "max_queue_us": max(
                        (compute_start_ns - item.enqueued_ns) // 1000
                        for item in active
                    ),
                    "status": "OK",
                })
                for item, result in zip(active, results):
                    item.future.set_result(result)
                active = []
        except BaseException as exc:
            with self._condition:
                self._error = exc
                for item in active:
                    if not item.future.done():
                        item.future.set_exception(exc)
                self._fail_all(exc)
                self._stopped = True
                self._condition.notify_all()

    def stop(self, timeout_s: float) -> None:
        if (
            type(timeout_s) not in (int, float)
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout must be positive")
        with self._condition:
            if self._stopping:
                raise CutBatcherError(f"{self.name} batcher stop repeated")
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError(f"{self.name} batcher did not stop")
        if self._error is not None:
            raise CutBatcherError(f"{self.name} batcher failed") from self._error
