#!/usr/bin/env python3
"""Bounded mixed decode/prefill batching for one fixed StageNet route."""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
if str(S22) not in sys.path:
    sys.path.insert(0, str(S22))

from stage_v3_client import BatchResult, BatchRow, ProtocolError


PHASE_DECODE = "decode"
PHASE_PREFILL = "prefill"
PHASES = (PHASE_DECODE, PHASE_PREFILL)


class BatchClient(Protocol):
    def batch(self, rows: Sequence[BatchRow]) -> Sequence[BatchResult]:
        ...


@dataclass(frozen=True)
class PhaseRow:
    row: BatchRow
    phase: str
    priority: int
    batch_wait_us: int | None = None


@dataclass
class _Pending:
    entry: PhaseRow
    future: Future[BatchResult]
    enqueued_ns: int
    latest_dispatch_ns: int
    order: int


class MixedPhaseBatcher:
    """Serialize one route while mixing old decode and new prefill rows."""

    MAX_PRIORITY = (1 << 31) - 1

    def __init__(
        self,
        name: str,
        client: BatchClient,
        max_rows: int,
        batch_knee: int,
        gather_us: int,
        queue_depth: int,
    ):
        for value in (max_rows, batch_knee, queue_depth):
            if type(value) is not int or value <= 0:
                raise ValueError("row and queue bounds must be positive integers")
        if batch_knee > max_rows:
            raise ValueError("batch knee exceeds route capacity")
        if type(gather_us) is not int or gather_us < 0:
            raise ValueError("gather time must be a nonnegative integer")
        if not name:
            raise ValueError("batcher name must be nonempty")

        self.name = name
        self.client = client
        self.max_rows = max_rows
        self.batch_knee = batch_knee
        self.gather_us = gather_us
        self.queue_depth = queue_depth
        self.events: list[dict] = []

        self._condition = threading.Condition()
        self._pending: list[_Pending] = []
        self._error: BaseException | None = None
        self._stopping = False
        self._stopped = False
        self._next_order = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"mixed-batch-{name}",
        )
        self._thread.start()

    @classmethod
    def _validate_entry(cls, entry: PhaseRow) -> None:
        if not isinstance(entry, PhaseRow):
            raise ValueError("submission must be a PhaseRow")
        if not isinstance(entry.row, BatchRow):
            raise ValueError("submission row must be a BatchRow")
        if entry.phase not in PHASES:
            raise ValueError("phase must be decode or prefill")
        if (
            type(entry.priority) is not int
            or not 0 <= entry.priority <= cls.MAX_PRIORITY
        ):
            raise ValueError("priority is out of range")
        if (
            entry.batch_wait_us is not None
            and (
                type(entry.batch_wait_us) is not int
                or entry.batch_wait_us < 0
            )
        ):
            raise ValueError("batch wait budget must be a nonnegative integer")

    def submit(
        self,
        entry: PhaseRow,
        timeout_s: float,
    ) -> Future[BatchResult]:
        return self.submit_many((entry,), timeout_s)[0]

    def submit_many(
        self,
        entries: Sequence[PhaseRow],
        timeout_s: float,
    ) -> tuple[Future[BatchResult], ...]:
        entries = tuple(entries)
        if not entries:
            raise ValueError("submission group must be nonempty")
        if len(entries) > self.queue_depth:
            raise ValueError("submission group exceeds queue capacity")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or timeout_s <= 0
        ):
            raise ValueError("submission timeout must be positive")
        for entry in entries:
            self._validate_entry(entry)

        identities = [
            (
                entry.row.request_id,
                entry.row.route_epoch,
                entry.row.seq_id,
                entry.row.position,
            )
            for entry in entries
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("submission group contains duplicate lineage")

        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while len(self._pending) + len(entries) > self.queue_depth:
                if self._error is not None:
                    raise RuntimeError(
                        f"{self.name} batcher failed",
                    ) from self._error
                if self._stopping:
                    raise RuntimeError(f"{self.name} batcher is stopping")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"{self.name} admission queue is full",
                    )
                self._condition.wait(remaining)

            if self._error is not None:
                raise RuntimeError(f"{self.name} batcher failed") from self._error
            if self._stopping:
                raise RuntimeError(f"{self.name} batcher is stopping")

            enqueued_ns = time.monotonic_ns()
            futures: list[Future[BatchResult]] = []
            for entry in entries:
                wait_us = (
                    self.gather_us
                    if entry.batch_wait_us is None
                    else min(self.gather_us, entry.batch_wait_us)
                )
                future: Future[BatchResult] = Future()
                self._pending.append(
                    _Pending(
                        entry=entry,
                        future=future,
                        enqueued_ns=enqueued_ns,
                        latest_dispatch_ns=enqueued_ns + wait_us * 1000,
                        order=self._next_order,
                    )
                )
                self._next_order += 1
                futures.append(future)
            self._condition.notify_all()
            return tuple(futures)

    @staticmethod
    def _phase_rank(pending: _Pending) -> int:
        return 0 if pending.entry.phase == PHASE_DECODE else 1

    @classmethod
    def _dispatch_order(cls, pending: _Pending) -> tuple[int, int, int, int]:
        return (
            cls._phase_rank(pending),
            pending.entry.priority,
            pending.latest_dispatch_ns,
            pending.order,
        )

    @staticmethod
    def _deadline_order(pending: _Pending) -> tuple[int, int, int]:
        return (
            pending.latest_dispatch_ns,
            pending.entry.priority,
            pending.order,
        )

    def _select_locked(
        self,
        now_ns: int,
    ) -> tuple[list[_Pending], str] | None:
        if not self._pending:
            return None
        if self._stopping:
            reason = "STOP_DRAIN"
        elif len(self._pending) >= self.batch_knee:
            reason = "BATCH_KNEE"
        elif min(item.latest_dispatch_ns for item in self._pending) <= now_ns:
            reason = "DEADLINE"
        else:
            return None

        overdue = sorted(
            (
                item
                for item in self._pending
                if item.latest_dispatch_ns <= now_ns
            ),
            key=self._deadline_order,
        )
        selected = overdue[: self.max_rows]
        selected_ids = {id(item) for item in selected}
        if len(selected) < self.max_rows:
            remainder = sorted(
                (
                    item
                    for item in self._pending
                    if id(item) not in selected_ids
                ),
                key=self._dispatch_order,
            )
            selected.extend(remainder[: self.max_rows - len(selected)])

        selected_ids = {id(item) for item in selected}
        self._pending = [
            item for item in self._pending if id(item) not in selected_ids
        ]
        selected.sort(key=self._dispatch_order)
        self._condition.notify_all()
        return selected, reason

    def _fail_pending_locked(self, error: BaseException) -> None:
        for pending in self._pending:
            if not pending.future.done():
                pending.future.set_exception(error)
        self._pending.clear()
        self._condition.notify_all()

    def _run(self) -> None:
        current: list[_Pending] = []
        try:
            while True:
                with self._condition:
                    while True:
                        if self._error is not None:
                            return
                        if self._stopping and not self._pending:
                            self._stopped = True
                            self._condition.notify_all()
                            return
                        now_ns = time.monotonic_ns()
                        selection = self._select_locked(now_ns)
                        if selection is not None:
                            current, release_reason = selection
                            break
                        if not self._pending:
                            self._condition.wait()
                        else:
                            deadline_ns = min(
                                item.latest_dispatch_ns
                                for item in self._pending
                            )
                            wait_s = max(
                                0.0,
                                (deadline_ns - now_ns) / 1e9,
                            )
                            self._condition.wait(wait_s)

                compute_start_ns = time.monotonic_ns()
                results = tuple(
                    self.client.batch(
                        [item.entry.row for item in current],
                    )
                )
                compute_end_ns = time.monotonic_ns()
                if len(results) != len(current):
                    raise ProtocolError(
                        f"{self.name} result count mismatch",
                    )

                self.events.append(
                    {
                        "batch_size": len(current),
                        "compute_us": (
                            compute_end_ns - compute_start_ns
                        ) // 1000,
                        "decode_rows": sum(
                            item.entry.phase == PHASE_DECODE
                            for item in current
                        ),
                        "max_queue_us": max(
                            (compute_start_ns - item.enqueued_ns) // 1000
                            for item in current
                        ),
                        "mixed_phase": len(
                            {item.entry.phase for item in current}
                        ) == 2,
                        "phases": [
                            item.entry.phase for item in current
                        ],
                        "positions": [
                            item.entry.row.position for item in current
                        ],
                        "prefill_rows": sum(
                            item.entry.phase == PHASE_PREFILL
                            for item in current
                        ),
                        "priorities": [
                            item.entry.priority for item in current
                        ],
                        "release_reason": release_reason,
                        "request_ids": [
                            item.entry.row.request_id for item in current
                        ],
                        "route_epochs": [
                            item.entry.row.route_epoch for item in current
                        ],
                        "sequence_ids": [
                            item.entry.row.seq_id for item in current
                        ],
                    }
                )
                for pending, result in zip(current, results):
                    pending.future.set_result(result)
                current = []
        except BaseException as error:
            with self._condition:
                self._error = error
                for pending in current:
                    if not pending.future.done():
                        pending.future.set_exception(error)
                self._fail_pending_locked(error)
                self._stopped = True
                self._condition.notify_all()

    def stop(self, timeout_s: float) -> None:
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or timeout_s <= 0
        ):
            raise ValueError("stop timeout must be positive")
        with self._condition:
            if self._stopping:
                raise RuntimeError(f"{self.name} batcher stop repeated")
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(float(timeout_s))
        if self._thread.is_alive():
            raise TimeoutError(f"{self.name} batcher did not stop")
        if self._error is not None:
            raise RuntimeError(f"{self.name} batcher failed") from self._error

    def abort(self, error: BaseException, timeout_s: float) -> None:
        with self._condition:
            if self._error is None:
                self._error = error
            self._stopping = True
            self._fail_pending_locked(self._error)
            self._condition.notify_all()
        self._thread.join(float(timeout_s))
        if self._thread.is_alive():
            raise TimeoutError(f"{self.name} batcher did not abort")
