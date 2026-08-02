#!/usr/bin/env python3

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
S24 = HERE.parent / "s24_overlap_handoff_poc"
S26 = HERE.parent / "s26_priority_scheduler"
for dependency in (HERE, S24, S26):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from physical_adapter import PhysicalRuntimeError, load_trace
from priority_shared_runtime import (
    EXPECTED_TRACE_HASH,
    prepare_trace,
    validate_priority_events,
)
from shared_physical_adapter import EXPECTED_ROUTES, validate_shared_pair


TRACE = S24 / "burstgpt-dense-mechanics.json"


def fake_routes(shared_tail: bool = True):
    common_client = object()
    common_slots = object()
    common_batcher = object()
    routes = []
    for route_id in ("R0", "R2"):
        stages = []
        for worker, start, end in EXPECTED_ROUTES[route_id]:
            is_tail = worker == "cuda-tail"
            stages.append(SimpleNamespace(
                worker_name=worker,
                layer_start=start,
                layer_end=end,
                client=(
                    common_client if is_tail else object()
                ),
                slots=(
                    common_slots if is_tail else object()
                ),
                batcher=(
                    common_batcher
                    if is_tail and (shared_tail or route_id == "R0")
                    else object()
                ),
            ))
        routes.append(SimpleNamespace(route_id=route_id, stages=tuple(stages)))
    return tuple(routes)


class TraceTests(unittest.TestCase):
    def test_frozen_dense_trace_is_accepted(self) -> None:
        trace = load_trace(TRACE)
        rows = prepare_trace(trace)
        self.assertEqual(trace["trace_hash"], EXPECTED_TRACE_HASH)
        self.assertEqual(len(rows), 60)
        self.assertEqual(
            [row.request_id for row in rows],
            sorted(row.request_id for row in rows),
        )

    def test_unknown_trace_is_rejected(self) -> None:
        trace = load_trace(TRACE)
        trace["trace_hash"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(PhysicalRuntimeError, "frozen dense"):
            prepare_trace(trace)

    def test_unmeasured_shape_is_rejected(self) -> None:
        trace = copy.deepcopy(load_trace(TRACE))
        trace["requests"][0]["output_steps"] = 5
        with self.assertRaisesRegex(PhysicalRuntimeError, "unmeasured"):
            prepare_trace(trace)


class TopologyTests(unittest.TestCase):
    def test_r0_and_r2_share_one_tail_queue(self) -> None:
        validate_shared_pair(fake_routes())

    def test_separate_tail_queue_is_rejected(self) -> None:
        with self.assertRaisesRegex(PhysicalRuntimeError, "share one"):
            validate_shared_pair(fake_routes(False))

    def test_route_shape_mutation_is_rejected(self) -> None:
        routes = list(fake_routes())
        first = routes[0]
        stages = list(first.stages)
        stages[1] = SimpleNamespace(
            **{**vars(stages[1]), "layer_end": 15},
        )
        routes[0] = SimpleNamespace(
            route_id=first.route_id, stages=tuple(stages),
        )
        with self.assertRaisesRegex(PhysicalRuntimeError, "measured route"):
            validate_shared_pair(tuple(routes))


class PriorityEvidenceTests(unittest.TestCase):
    def valid(self):
        return {
            "cuda-prefix": [{"priorities": [0], "status": "OK"}],
            "cuda-mid": [{"priorities": [0], "status": "OK"}],
            "op12-prefix": [{"priorities": [1, 2], "status": "OK"}],
            "op15-mid": [{"priorities": [1, 2], "status": "OK"}],
            "cuda-tail": [
                {"priorities": [0], "status": "OK"},
                {"priorities": [1, 2], "status": "OK"},
            ],
        }

    def test_valid_priority_evidence(self) -> None:
        validate_priority_events(self.valid())

    def test_urgent_background_mix_is_rejected(self) -> None:
        events = self.valid()
        events["cuda-tail"][0]["priorities"] = [0, 1]
        with self.assertRaisesRegex(PhysicalRuntimeError, "mixed urgent"):
            validate_priority_events(events)

    def test_missing_tail_evidence_is_rejected(self) -> None:
        events = self.valid()
        events["cuda-tail"] = []
        with self.assertRaisesRegex(PhysicalRuntimeError, "no batch"):
            validate_priority_events(events)

    def test_failed_batch_is_rejected(self) -> None:
        events = self.valid()
        events["cuda-tail"][0]["status"] = "ERROR"
        with self.assertRaisesRegex(PhysicalRuntimeError, "failed batch"):
            validate_priority_events(events)


if __name__ == "__main__":
    unittest.main()
