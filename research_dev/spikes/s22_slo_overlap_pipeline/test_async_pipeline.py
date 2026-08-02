#!/usr/bin/env python3

from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import Future

from async_pipeline import (
    DeviceBatcher,
    RequestSpec,
    parse_endpoint,
    parse_tokens,
    run_request,
    summarize_batches,
)
from stage_v3_client import BatchResult, BatchRow


class FakeClient:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.batch_sizes: list[int] = []
        self.batch_request_ids: list[list[int]] = []

    def batch(self, rows: list[BatchRow]) -> tuple[BatchResult, ...]:
        if self.fail:
            raise RuntimeError("injected failure")
        self.batch_sizes.append(len(rows))
        self.batch_request_ids.append([row.request_id for row in rows])
        return tuple(
            BatchResult(
                row.request_id, row.route_epoch, row.seq_id, row.position,
                (float(row.request_id),), None,
            )
            for row in rows
        )


class BlockingFirstClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def batch(self, rows: list[BatchRow]) -> tuple[BatchResult, ...]:
        if not self.batch_sizes:
            self.first_started.set()
            if not self.release_first.wait(timeout=2):
                raise TimeoutError("test did not release the first batch")
        return super().batch(rows)


class ImmediateBatcher:
    def __init__(self, terminal: bool):
        self.terminal = terminal
        self.rows: list[BatchRow] = []

    def submit(
        self, row: BatchRow, _timeout_s: float, _batch_wait_us: int | None = None,
    ) -> Future[BatchResult]:
        self.rows.append(row)
        future: Future[BatchResult] = Future()
        future.set_result(BatchResult(
            row.request_id, row.route_epoch, row.seq_id, row.position,
            None if self.terminal else (float(row.position),),
            100 + row.position if self.terminal else None,
        ))
        return future


class DeviceBatcherTests(unittest.TestCase):
    def test_gather_forms_one_batch(self) -> None:
        client = FakeClient()
        batcher = DeviceBatcher("test", client, 4, 50000, 8)  # type: ignore[arg-type]
        futures = [
            batcher.submit(BatchRow(index, 1, index, 0, 2), 1.0)
            for index in (1, 2, 3)
        ]
        results = [future.result(timeout=2) for future in futures]
        batcher.stop(2)
        self.assertEqual(client.batch_sizes, [3])
        self.assertEqual([result.request_id for result in results], [1, 2, 3])
        self.assertEqual(summarize_batches(batcher.events)["max_batch"], 3)

    def test_compute_failure_reaches_future_and_stop(self) -> None:
        batcher = DeviceBatcher(
            "test", FakeClient(fail=True), 1, 0, 2,  # type: ignore[arg-type]
        )
        future = batcher.submit(BatchRow(1, 1, 0, 0, 2), 1.0)
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            future.result(timeout=2)
        with self.assertRaisesRegex(RuntimeError, "batcher failed"):
            batcher.stop(2)

    def test_stop_excludes_later_admission(self) -> None:
        client = FakeClient()
        batcher = DeviceBatcher("test", client, 1, 0, 2)  # type: ignore[arg-type]
        batcher.stop(2)
        with self.assertRaisesRegex(RuntimeError, "stopping"):
            batcher.submit(BatchRow(1, 1, 0, 0, 2), 1.0)

    def test_submit_stop_race_conserves_accepted_future(self) -> None:
        client = FakeClient()
        batcher = DeviceBatcher("test", client, 1, 1000, 16)  # type: ignore[arg-type]
        gate = threading.Barrier(2)
        accepted = []
        rejected = []

        def submit() -> None:
            gate.wait()
            try:
                accepted.append(batcher.submit(BatchRow(1, 1, 0, 0, 2), 1.0))
            except RuntimeError:
                rejected.append(True)

        thread = threading.Thread(target=submit)
        thread.start()
        gate.wait()
        batcher.stop(2)
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(accepted) + len(rejected), 1)
        if accepted:
            self.assertEqual(accepted[0].result(timeout=2).request_id, 1)

    def test_zero_wait_budget_prevents_gathering(self) -> None:
        client = BlockingFirstClient()
        batcher = DeviceBatcher("test", client, 4, 50000, 8)  # type: ignore[arg-type]
        first = batcher.submit(BatchRow(1, 1, 0, 0, 2), 1.0, 0)
        self.assertTrue(client.first_started.wait(timeout=2))
        second = batcher.submit(BatchRow(2, 1, 1, 0, 2), 1.0, 0)
        client.release_first.set()
        first.result(timeout=2)
        second.result(timeout=2)
        batcher.stop(2)
        self.assertEqual(client.batch_sizes, [1, 1])

    def test_overdue_backlog_is_drained_without_waiting(self) -> None:
        client = BlockingFirstClient()
        batcher = DeviceBatcher("test", client, 4, 0, 8)  # type: ignore[arg-type]
        first = batcher.submit(BatchRow(1, 1, 0, 0, 2), 1.0, 0)
        self.assertTrue(client.first_started.wait(timeout=2))
        second = batcher.submit(BatchRow(2, 1, 1, 0, 2), 1.0, 0)
        third = batcher.submit(BatchRow(3, 1, 2, 0, 2), 1.0, 0)
        time.sleep(0.01)
        client.release_first.set()
        for future in (first, second, third):
            future.result(timeout=2)
        batcher.stop(2)
        self.assertEqual(client.batch_sizes, [1, 2])

    def test_urgent_row_overtakes_background_backlog(self) -> None:
        client = BlockingFirstClient()
        batcher = DeviceBatcher("test", client, 2, 0, 8)  # type: ignore[arg-type]
        first = batcher.submit(BatchRow(1, 1, 0, 0, 2), 1.0, 0, 2)
        self.assertTrue(client.first_started.wait(timeout=2))
        low_a = batcher.submit(BatchRow(2, 1, 1, 0, 2), 1.0, 0, 2)
        low_b = batcher.submit(BatchRow(3, 1, 2, 0, 2), 1.0, 0, 1)
        urgent = batcher.submit(BatchRow(4, 1, 3, 0, 2), 1.0, 0, 0)
        client.release_first.set()
        for future in (first, low_a, low_b, urgent):
            future.result(timeout=2)
        batcher.stop(2)
        self.assertEqual(client.batch_request_ids, [[1], [4], [3, 2]])

    def test_background_priority_classes_can_merge(self) -> None:
        client = FakeClient()
        batcher = DeviceBatcher("test", client, 2, 50000, 8)  # type: ignore[arg-type]
        high = batcher.submit(BatchRow(1, 1, 0, 0, 2), 1.0, None, 1)
        low = batcher.submit(BatchRow(2, 1, 1, 0, 2), 1.0, None, 2)
        high.result(timeout=2)
        low.result(timeout=2)
        batcher.stop(2)
        self.assertEqual(client.batch_request_ids, [[1, 2]])

    def test_invalid_priority_is_rejected(self) -> None:
        batcher = DeviceBatcher("test", FakeClient(), 1, 0, 2)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "priority"):
            batcher.submit(BatchRow(1, 1, 0, 0, 2), 1.0, 0, True)
        batcher.stop(2)

    def test_endpoint_parser(self) -> None:
        self.assertEqual(parse_endpoint("127.0.0.1:9000"), ("127.0.0.1", 9000))
        with self.assertRaises(Exception):
            parse_endpoint("bad")

    def test_multi_token_prefill_then_decode(self) -> None:
        head = ImmediateBatcher(False)
        tail = ImmediateBatcher(True)
        result = run_request(
            RequestSpec(1, 1, "head", 0, 0, 2, 1000.0, 0, (2, 3, 4)),
            head,  # type: ignore[arg-type]
            tail,  # type: ignore[arg-type]
            2, threading.Barrier(1), 1.0,
        )
        self.assertEqual(result["prompt_length"], 3)
        self.assertEqual(result["tokens"], [102, 103])
        self.assertEqual(
            [(row.position, row.token) for row in head.rows],
            [(0, 2), (1, 3), (2, 4), (3, 102)],
        )

    def test_token_parser(self) -> None:
        self.assertEqual(parse_tokens("2,3,4"), (2, 3, 4))
        with self.assertRaises(Exception):
            parse_tokens("")


if __name__ == "__main__":
    unittest.main()
