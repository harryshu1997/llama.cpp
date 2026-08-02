#!/usr/bin/env python3

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
S22 = ROOT.parent / "s22_slo_overlap_pipeline"
S23 = ROOT.parent / "s23_dense_trace_runtime"
for dependency in (ROOT, S23, S22):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from cut_batcher import CutBatcher, LockedStageClient  # noqa: E402
from dynamic_route_runtime import (  # noqa: E402
    DynamicRequest,
    DynamicRoute,
    DynamicRouteError,
    DynamicRouteRunner,
    DynamicStage,
)
from runtime_support import SequenceSlotPool  # noqa: E402
from stage_v3_client import BatchResult, Hello  # noqa: E402


class FakeStageClient:
    def __init__(self, terminal: bool) -> None:
        self.terminal = terminal
        self.removed: list[tuple[int, int, int]] = []
        self.batch_sizes: list[int] = []

    def range_batch(self, rows, layer_start: int, layer_end: int):
        del layer_start, layer_end
        self.batch_sizes.append(len(rows))
        return tuple(
            BatchResult(
                row.request_id,
                row.route_epoch,
                row.seq_id,
                row.position,
                None if self.terminal else (float(row.token), 1.0),
                row.token + 1 if self.terminal else None,
            )
            for row in rows
        )

    def remove(self, seq_id: int, request_id: int, route_epoch: int):
        self.removed.append((seq_id, request_id, route_epoch))
        return object()


def hello(start: int, end: int, terminal: bool) -> Hello:
    capabilities = 0x0F | 0x40 | (0x10 if terminal else 0)
    return Hello(start, end, 48, 2, 8, 64, 64, 64, capabilities, 7, "0" * 64)


class RuntimeFixture:
    def __init__(self, knee: int = 1, gather_us: int = 0) -> None:
        self.head_raw = FakeStageClient(False)
        self.tail_raw = FakeStageClient(True)
        self.head_client = LockedStageClient(self.head_raw)
        self.tail_client = LockedStageClient(self.tail_raw)
        self.head_batcher = CutBatcher(
            "head", self.head_client, {4: (0, 4), 8: (0, 8)},
            {4: knee, 8: knee}, 64, gather_us, 64,
        )
        self.tail_batcher = CutBatcher(
            "tail", self.tail_client, {4: (4, 48), 8: (8, 48)},
            {4: knee, 8: knee}, 64, gather_us, 64,
        )
        self.head = DynamicStage(
            "head", self.head_client, hello(0, 8, False),
            SequenceSlotPool(8), self.head_batcher, False,
        )
        self.tail = DynamicStage(
            "tail", self.tail_client, hello(4, 48, True),
            SequenceSlotPool(8), self.tail_batcher, True,
        )
        self.runner = DynamicRouteRunner((
            DynamicRoute("head-c4", self.head, self.tail, 4),
            DynamicRoute("head-c8", self.head, self.tail, 8),
        ))

    def stop(self) -> None:
        self.head_batcher.stop(1.0)
        self.tail_batcher.stop(1.0)


class DynamicRouteRuntimeTests(unittest.TestCase):
    def test_complete_route_preserves_cut_and_cleans_state(self) -> None:
        fixture = RuntimeFixture()
        try:
            request = DynamicRequest(
                10, 1, "head-c8", (2, 2, 2, 2), 3, 1, 1_000_000, 0,
            )
            arrival = time.monotonic_ns()
            outcome = fixture.runner.run(request, 1.0, arrival)
            self.assertEqual(outcome.cut, 8)
            self.assertEqual(outcome.output_tokens, (3, 4, 5))
            self.assertEqual(fixture.runner.pins(), {})
            self.assertEqual(fixture.head.slots.leased(), {})
            self.assertEqual(fixture.tail.slots.leased(), {})
            self.assertEqual(len(fixture.head_raw.removed), 1)
            self.assertEqual(len(fixture.tail_raw.removed), 1)
            self.assertTrue(all(event["cut"] == 8 for event in fixture.head_batcher.events))
            self.assertTrue(all(event["cut"] == 8 for event in fixture.tail_batcher.events))
        finally:
            fixture.stop()

    def test_unknown_route_fails_before_lease(self) -> None:
        fixture = RuntimeFixture()
        try:
            request = DynamicRequest(
                10, 1, "unknown", (2,), 1, 1, 1_000_000, 0,
            )
            with self.assertRaises(DynamicRouteError):
                fixture.runner.run(request, 1.0)
            self.assertEqual(fixture.runner.pins(), {})
        finally:
            fixture.stop()

    def test_prefill_is_released_in_bounded_quanta(self) -> None:
        fixture = RuntimeFixture(knee=4, gather_us=100_000)
        try:
            request = DynamicRequest(
                10, 1, "head-c8", tuple(range(8)), 1, 1, 1_000_000, 100_000,
                prefill_quantum=4,
            )
            outcome = fixture.runner.run(request, 1.0)
            self.assertEqual(outcome.output_tokens, (8,))
            self.assertEqual(fixture.head_raw.batch_sizes, [4, 4])
            self.assertEqual(fixture.tail_raw.batch_sizes, [4, 4])
        finally:
            fixture.stop()

    def test_stop_token_ends_decode_before_budget(self) -> None:
        fixture = RuntimeFixture()
        try:
            request = DynamicRequest(
                10, 1, "head-c8", (2, 2), 8, 1, 1_000_000, 0,
                stop_tokens=(3,),
            )
            outcome = fixture.runner.run(request, 1.0)
            self.assertEqual(outcome.output_tokens, (3,))
            self.assertEqual(outcome.finish_reason, "stop")
        finally:
            fixture.stop()

    def test_route_requires_exact_handoff(self) -> None:
        fixture = RuntimeFixture()
        try:
            bad_tail_batcher = CutBatcher(
                "bad-tail", fixture.tail_client, {4: (4, 47)}, {4: 1},
                64, 0, 8,
            )
            bad_tail = DynamicStage(
                "bad-tail", fixture.tail_client, hello(4, 48, True),
                SequenceSlotPool(8), bad_tail_batcher, True,
            )
            with self.assertRaises(ValueError):
                DynamicRoute("bad", fixture.head, bad_tail, 4)
            bad_tail_batcher.stop(1.0)
        finally:
            fixture.stop()

    def test_boolean_request_fields_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DynamicRequest(True, 1, "head-c4", (2,), 1, 1, 1, 0)
        with self.assertRaises(ValueError):
            DynamicRequest(1, 1, "head-c4", (2,), 1, 1, 1, 0, True)
        with self.assertRaises(ValueError):
            DynamicRequest(1, 1, "head-c4", (2,), 1, 1, 1, 0, stop_tokens=(3, 3))


if __name__ == "__main__":
    unittest.main()
