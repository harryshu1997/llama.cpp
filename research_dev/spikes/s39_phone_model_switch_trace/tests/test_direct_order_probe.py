#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
S22 = S39.parent / "s22_slo_overlap_pipeline"
for path in (S39, S22):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from direct_order_probe import (
    REQUEST_BASE,
    expected_batch_sizes,
    sequence_order,
    validate_order_events,
)
from stage_v3_client import ProtocolError


def valid_events(order_name: str) -> list[dict]:
    order = sequence_order(order_name)
    events = []
    for index, size in enumerate([160] + [32] * 7):
        prefill = index == 0
        sequence_ids = (
            [sequence_id for sequence_id in order for _ in range(5)]
            if prefill
            else order
        )
        events.append(
            {
                "batch_size": size,
                "decode_rows": 0 if prefill else 32,
                "mixed_phase": False,
                "phases": ["prefill" if prefill else "decode"] * size,
                "positions": (
                    list(range(5)) * 32
                    if prefill
                    else [4 + index] * 32
                ),
                "prefill_rows": size if prefill else 0,
                "release_reason": "BATCH_KNEE" if prefill else "DEADLINE",
                "request_ids": (
                    [
                        REQUEST_BASE + sequence_id
                        for sequence_id in order
                        for _ in range(5)
                    ]
                    if prefill
                    else [REQUEST_BASE + sequence_id for sequence_id in order]
                ),
                "sequence_ids": sequence_ids,
            }
        )
    return events


class DirectOrderProbeTests(unittest.TestCase):
    def test_frozen_orders_are_distinct_permutations(self):
        sorted_order = sequence_order("sorted")
        shuffled_order = sequence_order("shuffled")
        self.assertEqual(sorted_order, list(range(32)))
        self.assertEqual(sorted(shuffled_order), sorted_order)
        self.assertNotEqual(shuffled_order, sorted_order)
        self.assertEqual(shuffled_order[:4], [11, 28, 13, 30])

    def test_expected_shapes(self):
        self.assertEqual(
            expected_batch_sizes(32, 5, 8),
            [160, 32, 32, 32, 32, 32, 32, 32],
        )

    def test_decode_reaches_small_batch_knee(self):
        events = valid_events("sorted")
        for item in events:
            item["release_reason"] = "BATCH_KNEE"
        validate_order_events(
            events,
            sequence_order("sorted"),
            5,
            8,
            32,
        )

    def test_sorted_plan_passes(self):
        validate_order_events(
            valid_events("sorted"),
            sequence_order("sorted"),
            5,
            8,
        )

    def test_shuffled_plan_passes(self):
        validate_order_events(
            valid_events("shuffled"),
            sequence_order("shuffled"),
            5,
            8,
        )

    def test_order_mutation_is_rejected(self):
        events = valid_events("shuffled")
        events[2]["sequence_ids"][0] = 0
        with self.assertRaisesRegex(ProtocolError, "physical row order"):
            validate_order_events(
                events,
                sequence_order("shuffled"),
                5,
                8,
            )

    def test_batch_mutation_is_rejected(self):
        events = valid_events("sorted")
        events[0]["batch_size"] = 159
        with self.assertRaisesRegex(ProtocolError, "batch sequence"):
            validate_order_events(
                events,
                sequence_order("sorted"),
                5,
                8,
            )

    def test_unknown_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sorted or shuffled"):
            sequence_order("unknown")


if __name__ == "__main__":
    unittest.main()
