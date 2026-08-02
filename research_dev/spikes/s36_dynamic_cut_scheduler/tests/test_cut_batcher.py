#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cut_batcher import (  # noqa: E402
    CutBatcher,
    CutBatcherError,
    LockedStageClient,
    RequestCutTable,
)


@dataclass(frozen=True)
class Row:
    request_id: int
    route_epoch: int
    seq_id: int
    position: int


class FakeClient:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.calls: list[tuple[tuple[Row, ...], int, int]] = []
        self.failure = failure
        self.lock = threading.Lock()

    def range_batch(
        self, rows: list[Row], layer_start: int, layer_end: int,
    ) -> tuple[Row, ...]:
        with self.lock:
            self.calls.append((tuple(rows), layer_start, layer_end))
        if self.failure is not None:
            raise self.failure
        return tuple(rows)


def make_batcher(
    client: FakeClient,
    knees: dict[int, int] | None = None,
    gather_us: int = 50_000,
) -> CutBatcher:
    return CutBatcher(
        "fake",
        LockedStageClient(client),
        {4: (0, 4), 8: (0, 8)},
        knees or {4: 2, 8: 2},
        max_rows=8,
        gather_us=gather_us,
        queue_depth=16,
    )


class CutBatcherTests(unittest.TestCase):
    def test_same_cut_mixes_prefill_and_decode_at_knee(self) -> None:
        client = FakeClient()
        batcher = make_batcher(client)
        first = batcher.submit(Row(1, 1, 0, 0), 4, "prefill", 1.0, priority=1)
        second = batcher.submit(Row(2, 1, 1, 4), 4, "decode", 1.0, priority=2)
        self.assertEqual(first.result(1.0).request_id, 1)
        self.assertEqual(second.result(1.0).request_id, 2)
        batcher.stop(1.0)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][1:], (0, 4))
        self.assertTrue(batcher.events[0]["mixed_phase"])
        self.assertEqual(batcher.events[0]["release_reason"], "BATCH_KNEE")

    def test_different_cuts_never_share_a_call(self) -> None:
        client = FakeClient()
        batcher = make_batcher(client, {4: 1, 8: 1})
        futures = [
            batcher.submit(Row(1, 1, 0, 0), 4, "prefill", 1.0, priority=1),
            batcher.submit(Row(2, 1, 1, 0), 8, "prefill", 1.0, priority=1),
        ]
        for future in futures:
            future.result(1.0)
        batcher.stop(1.0)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual({call[2] for call in client.calls}, {4, 8})
        self.assertTrue(all(len(call[0]) == 1 for call in client.calls))

    def test_priority_zero_never_mixes_with_background(self) -> None:
        client = FakeClient()
        batcher = make_batcher(client, gather_us=200_000)
        urgent = batcher.submit(
            Row(1, 1, 0, 0), 4, "prefill", 1.0,
            batch_wait_us=0, priority=0,
        )
        background = batcher.submit(
            Row(2, 1, 1, 0), 4, "prefill", 1.0,
            batch_wait_us=0, priority=1,
        )
        urgent.result(1.0)
        background.result(1.0)
        batcher.stop(1.0)
        self.assertEqual(len(client.calls), 2)
        for event in batcher.events:
            self.assertFalse(0 in event["priorities"] and len(set(event["priorities"])) > 1)

    def test_latest_safe_start_releases_below_knee(self) -> None:
        client = FakeClient()
        batcher = make_batcher(client, {4: 4, 8: 4}, gather_us=500_000)
        future = batcher.submit(
            Row(1, 1, 0, 0), 4, "decode", 1.0,
            batch_wait_us=1_000, priority=1,
        )
        future.result(1.0)
        batcher.stop(1.0)
        self.assertEqual(batcher.events[0]["release_reason"], "LATEST_SAFE_START")
        self.assertGreaterEqual(batcher.events[0]["max_queue_us"], 500)

    def test_stop_drains_pending_rows(self) -> None:
        client = FakeClient()
        batcher = make_batcher(client, {4: 8, 8: 8}, gather_us=5_000_000)
        future = batcher.submit(Row(1, 1, 0, 0), 8, "prefill", 1.0, priority=2)
        batcher.stop(1.0)
        self.assertEqual(future.result(0).request_id, 1)
        self.assertEqual(batcher.events[0]["release_reason"], "DRAIN")

    def test_worker_failure_reaches_all_pending_futures(self) -> None:
        client = FakeClient(RuntimeError("compute failed"))
        batcher = make_batcher(client, {4: 2, 8: 2})
        first = batcher.submit(Row(1, 1, 0, 0), 4, "prefill", 1.0, priority=1)
        second = batcher.submit(Row(2, 1, 1, 0), 4, "decode", 1.0, priority=1)
        with self.assertRaisesRegex(RuntimeError, "compute failed"):
            first.result(1.0)
        with self.assertRaisesRegex(RuntimeError, "compute failed"):
            second.result(1.0)
        with self.assertRaises(CutBatcherError):
            batcher.submit(Row(3, 1, 2, 0), 4, "prefill", 1.0)
        with self.assertRaises(CutBatcherError):
            batcher.stop(1.0)

    def test_submission_validation(self) -> None:
        batcher = make_batcher(FakeClient())
        row = Row(1, 1, 0, 0)
        with self.assertRaises(ValueError):
            batcher.submit(row, 6, "prefill", 1.0)
        with self.assertRaises(ValueError):
            batcher.submit(row, 4, "other", 1.0)
        with self.assertRaises(ValueError):
            batcher.submit(row, 4, "prefill", 1.0, priority=True)
        with self.assertRaises(ValueError):
            batcher.submit(row, 4, "prefill", 1.0, batch_wait_us=-1)
        with self.assertRaises(ValueError):
            batcher.submit(row, 4, "prefill", math.inf)
        batcher.stop(1.0)


class RequestCutTableTests(unittest.TestCase):
    def test_cut_is_pinned_until_remove(self) -> None:
        table = RequestCutTable()
        table.pin(1, 3, 4)
        table.pin(1, 3, 4)
        table.require(1, 3, 4)
        with self.assertRaises(CutBatcherError):
            table.pin(1, 3, 8)
        table.remove(1, 3)
        table.pin(1, 4, 8)
        self.assertEqual(table.snapshot(), {(1, 4): 8})

    def test_unknown_cut_lease_is_rejected(self) -> None:
        table = RequestCutTable()
        with self.assertRaises(CutBatcherError):
            table.require(1, 1, 4)
        with self.assertRaises(CutBatcherError):
            table.remove(1, 1)


if __name__ == "__main__":
    unittest.main()
