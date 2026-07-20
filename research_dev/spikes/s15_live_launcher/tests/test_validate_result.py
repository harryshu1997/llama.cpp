#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
LIVE = HERE.parent
sys.path.insert(0, str(LIVE))

import validate_result as validator  # noqa: E402


COHORT = json.loads(
    (LIVE.parent / "s15_burst_cohort/cohort.json").read_text(encoding="ascii")
)


def decisions() -> list[dict]:
    values = []
    latest_safe = COHORT["admission_schedule"]["latest_safe_launch_us"]
    requests = COHORT["requests"]
    for request in requests[:-1]:
        values.append({
            "after_request_id": request["event_id"],
            "at_us": request["observed_t_us"],
            "action": "WAIT",
            "reason": "bounded_batch_wait",
            "next_wake_us": latest_safe,
        })
    values.append({
        "after_request_id": requests[-1]["event_id"],
        "at_us": requests[-1]["observed_t_us"],
        "action": "LAUNCH",
        "reason": "target_batch_ready",
        "batch_size": 32,
        "request_ids": [request["event_id"] for request in requests],
    })
    return values


class ReplayTimingTests(unittest.TestCase):
    def test_observed_transport_fits_exact_remaining_budget(self) -> None:
        self.assertEqual(
            validator.validate_replay_timing(COHORT, decisions(), 3_940_857),
            59_143,
        )

    def test_five_second_transport_is_not_the_launch_budget(self) -> None:
        with self.assertRaisesRegex(validator.ValidationError, "misses.*deadline"):
            validator.validate_replay_timing(COHORT, decisions(), 4_000_001)

    def test_wait_timestamp_mutation_is_rejected(self) -> None:
        mutated = copy.deepcopy(decisions())
        mutated[0]["at_us"] += 1
        with self.assertRaisesRegex(validator.ValidationError, "frozen arrival replay"):
            validator.validate_replay_timing(COHORT, mutated, 3_940_857)

    def test_wait_wake_mutation_is_rejected(self) -> None:
        mutated = copy.deepcopy(decisions())
        mutated[0]["next_wake_us"] += 1
        with self.assertRaisesRegex(validator.ValidationError, "frozen arrival replay"):
            validator.validate_replay_timing(COHORT, mutated, 3_940_857)

    def test_reported_deadline_must_be_derived(self) -> None:
        mutated = copy.deepcopy(COHORT)
        mutated["admission_schedule"]["earliest_deadline_us"] += 1
        with self.assertRaisesRegex(validator.ValidationError, "earliest deadline"):
            validator.validate_replay_timing(mutated, decisions(), 3_940_857)


if __name__ == "__main__":
    unittest.main()
