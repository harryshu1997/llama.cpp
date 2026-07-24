#!/usr/bin/env python3

import sys
import time
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
S22 = S39.parent / "s22_slo_overlap_pipeline"
for path in (S39, S22):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mixed_phase_batcher import (
    PHASE_DECODE,
    PHASE_PREFILL,
    MixedPhaseBatcher,
    PhaseRow,
)
from stage_v3_client import BatchResult, BatchRow


def row(request_id: int, seq_id: int, position: int) -> BatchRow:
    return BatchRow(request_id, 1, seq_id, position, request_id + position)


class FakeClient:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[list[BatchRow]] = []

    def batch(self, rows):
        rows = list(rows)
        self.calls.append(rows)
        if self.fail:
            raise RuntimeError("injected compute failure")
        return tuple(
            BatchResult(
                item.request_id,
                item.route_epoch,
                item.seq_id,
                item.position,
                None,
                item.token + 1,
            )
            for item in rows
        )


class MixedPhaseBatcherTests(unittest.TestCase):
    def test_mixed_knee_orders_decode_before_prefill(self):
        client = FakeClient()
        batcher = MixedPhaseBatcher("test", client, 8, 4, 100000, 8)
        entries = (
            PhaseRow(row(20, 1, 0), PHASE_PREFILL, 2),
            PhaseRow(row(21, 2, 1), PHASE_PREFILL, 2),
            PhaseRow(row(10, 0, 5), PHASE_DECODE, 0),
            PhaseRow(row(22, 3, 2), PHASE_PREFILL, 2),
        )
        futures = batcher.submit_many(entries, 1.0)
        results = tuple(future.result(timeout=1.0) for future in futures)
        batcher.stop(1.0)

        self.assertEqual([result.request_id for result in results], [20, 21, 10, 22])
        self.assertEqual(
            [item.request_id for item in client.calls[0]],
            [10, 20, 21, 22],
        )
        self.assertEqual(batcher.events[0]["release_reason"], "BATCH_KNEE")
        self.assertEqual(
            batcher.events[0]["phases"],
            [PHASE_DECODE, PHASE_PREFILL, PHASE_PREFILL, PHASE_PREFILL],
        )
        self.assertTrue(batcher.events[0]["mixed_phase"])

    def test_deadline_releases_partial_batch(self):
        client = FakeClient()
        batcher = MixedPhaseBatcher("test", client, 8, 8, 1000, 8)
        future = batcher.submit(
            PhaseRow(row(10, 0, 5), PHASE_DECODE, 0),
            1.0,
        )
        self.assertEqual(future.result(timeout=1.0).token, 16)
        batcher.stop(1.0)
        self.assertEqual(batcher.events[0]["batch_size"], 1)
        self.assertEqual(batcher.events[0]["release_reason"], "DEADLINE")

    def test_stop_drains_accepted_rows(self):
        client = FakeClient()
        batcher = MixedPhaseBatcher("test", client, 8, 8, 1000000, 8)
        future = batcher.submit(
            PhaseRow(row(10, 0, 5), PHASE_DECODE, 0),
            1.0,
        )
        batcher.stop(1.0)
        self.assertEqual(future.result(timeout=1.0).token, 16)
        self.assertEqual(batcher.events[0]["release_reason"], "STOP_DRAIN")

    def test_group_capacity_rejects_without_partial_admission(self):
        client = FakeClient()
        batcher = MixedPhaseBatcher("test", client, 8, 8, 100000, 4)
        entries = tuple(
            PhaseRow(row(10 + index, index, 0), PHASE_PREFILL, 2)
            for index in range(5)
        )
        with self.assertRaisesRegex(ValueError, "queue capacity"):
            batcher.submit_many(entries, 1.0)
        batcher.stop(1.0)
        self.assertEqual(client.calls, [])
        self.assertEqual(batcher.events, [])

    def test_invalid_phase_priority_and_wait_are_rejected(self):
        client = FakeClient()
        batcher = MixedPhaseBatcher("test", client, 8, 8, 100000, 8)
        bad = (
            PhaseRow(row(1, 0, 0), "other", 0),
            PhaseRow(row(1, 0, 0), PHASE_PREFILL, True),
            PhaseRow(row(1, 0, 0), PHASE_PREFILL, 0, True),
        )
        for entry in bad:
            with self.assertRaises(ValueError):
                batcher.submit(entry, 1.0)
        batcher.stop(1.0)

    def test_duplicate_lineage_is_rejected(self):
        client = FakeClient()
        batcher = MixedPhaseBatcher("test", client, 8, 8, 100000, 8)
        duplicate = PhaseRow(row(1, 0, 0), PHASE_PREFILL, 2)
        with self.assertRaisesRegex(ValueError, "duplicate lineage"):
            batcher.submit_many((duplicate, duplicate), 1.0)
        batcher.stop(1.0)

    def test_compute_failure_reaches_future_and_stop(self):
        client = FakeClient(fail=True)
        batcher = MixedPhaseBatcher("test", client, 8, 1, 100000, 8)
        future = batcher.submit(
            PhaseRow(row(10, 0, 5), PHASE_DECODE, 0),
            1.0,
        )
        with self.assertRaisesRegex(RuntimeError, "injected compute failure"):
            future.result(timeout=1.0)
        with self.assertRaisesRegex(RuntimeError, "batcher failed"):
            batcher.stop(1.0)

    def test_submission_timeout_type_is_fail_closed(self):
        client = FakeClient()
        batcher = MixedPhaseBatcher("test", client, 8, 8, 100000, 8)
        entry = PhaseRow(row(1, 0, 0), PHASE_PREFILL, 2)
        with self.assertRaises(ValueError):
            batcher.submit(entry, True)
        batcher.stop(1.0)


if __name__ == "__main__":
    unittest.main()
