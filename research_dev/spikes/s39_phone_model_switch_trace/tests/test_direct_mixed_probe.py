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

from direct_mixed_probe import (
    NEW_REQUEST_BASE,
    SEED_REQUEST_BASE,
    expected_batch_sizes,
    validate_mixed_events,
)
from mixed_phase_batcher import PHASE_DECODE, PHASE_PREFILL
from stage_v3_client import ProtocolError


def valid_events() -> list[dict]:
    events = [
        {
            "batch_size": 80,
            "decode_rows": 0,
            "mixed_phase": False,
            "phases": [PHASE_PREFILL] * 80,
            "positions": list(range(5)) * 16,
            "prefill_rows": 80,
            "release_reason": "DEADLINE",
            "request_ids": [
                SEED_REQUEST_BASE + index
                for index in range(16)
                for _ in range(5)
            ],
        },
        {
            "batch_size": 96,
            "decode_rows": 16,
            "mixed_phase": True,
            "phases": [PHASE_DECODE] * 16 + [PHASE_PREFILL] * 80,
            "positions": [5] * 16 + list(range(5)) * 16,
            "prefill_rows": 80,
            "release_reason": "BATCH_KNEE",
            "request_ids": (
                [SEED_REQUEST_BASE + index for index in range(16)]
                + [
                    NEW_REQUEST_BASE + index
                    for index in range(16)
                    for _ in range(5)
                ]
            ),
        },
    ]
    for _ in range(6):
        events.append(
            {
                "batch_size": 32,
                "decode_rows": 32,
                "mixed_phase": False,
                "phases": [PHASE_DECODE] * 32,
                "positions": [],
                "prefill_rows": 0,
                "release_reason": "DEADLINE",
                "request_ids": [],
            }
        )
    events.append(
        {
            "batch_size": 16,
            "decode_rows": 16,
            "mixed_phase": False,
            "phases": [PHASE_DECODE] * 16,
            "positions": [],
            "prefill_rows": 0,
            "release_reason": "DEADLINE",
            "request_ids": [],
        }
    )
    return events


class DirectMixedProbeTests(unittest.TestCase):
    def test_expected_plan(self):
        self.assertEqual(
            expected_batch_sizes(16, 5, 8),
            [80, 96, 32, 32, 32, 32, 32, 32, 16],
        )
        result = validate_mixed_events(valid_events(), 16, 5, 8)
        self.assertEqual(result["batch_index"], 1)
        self.assertEqual(result["decode_rows"], 16)
        self.assertEqual(result["prefill_rows"], 80)

    def test_wrong_physical_phase_order_is_rejected(self):
        events = valid_events()
        events[1]["phases"][0] = PHASE_PREFILL
        with self.assertRaisesRegex(ProtocolError, "phase ordering"):
            validate_mixed_events(events, 16, 5, 8)

    def test_wrong_mixed_position_is_rejected(self):
        events = valid_events()
        events[1]["positions"][16] = 5
        with self.assertRaisesRegex(ProtocolError, "phase ordering"):
            validate_mixed_events(events, 16, 5, 8)

    def test_extra_mixed_event_is_rejected(self):
        events = valid_events()
        events[2]["mixed_phase"] = True
        with self.assertRaisesRegex(ProtocolError, "exactly one"):
            validate_mixed_events(events, 16, 5, 8)

    def test_batch_size_mutation_is_rejected(self):
        events = valid_events()
        events[-1]["batch_size"] = 17
        with self.assertRaisesRegex(ProtocolError, "batch sequence"):
            validate_mixed_events(events, 16, 5, 8)


if __name__ == "__main__":
    unittest.main()
