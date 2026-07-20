#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
LIVE = HERE.parent
S15 = LIVE.parent / "s15_runtime_dispatch"
S14 = LIVE.parent / "s14_mixed_streaming_scheduler"
sys.path[:0] = [str(LIVE), str(S15), str(S14)]

from executor_contract import RecordedExecutor, RecordedOutcome, recorded_key  # noqa: E402
from power_frontier_policy import BatchDecision, WorkItem  # noqa: E402
from priority_batch_runtime import Launch  # noqa: E402
from route_fixtures import op15_b32_snapshot  # noqa: E402
from route_registry import ReadyRouteRegistry  # noqa: E402
from runtime_dispatch import LaneBinding, MixedDispatchCoordinator  # noqa: E402
import physical_launcher as launcher  # noqa: E402


class CohortDispatchTests(unittest.TestCase):
    def test_31_waits_then_exact_b32_launch(self) -> None:
        cohort = json.loads(launcher.COHORT_PATH.read_text(encoding="ascii"))
        request_ids = tuple(value["event_id"] for value in cohort["requests"])
        snapshot = op15_b32_snapshot(route_epoch=12)
        recorded = RecordedOutcome(
            "completed", 2_900_000, snapshot.profile_id, snapshot.route_epoch,
            snapshot.residency_epoch, snapshot.device_boot_epoch,
            launcher.EXPECTED_COHORT, launcher.EXPECTED_INPUT,
        )
        registry = ReadyRouteRegistry()
        registry.install([snapshot])
        coordinator = MixedDispatchCoordinator(
            registry,
            RecordedExecutor({recorded_key(snapshot.route_id, request_ids): recorded}),
            (LaneBinding(
                "op15", snapshot.route_id, "phone", 5_000_000,
                launcher.EXPECTED_COHORT, launcher.EXPECTED_INPUT,
            ),),
            queue_capacity=32,
        )
        decisions = []
        for request in cohort["requests"]:
            now_us = request["observed_t_us"]
            item = WorkItem(
                request["event_id"], "generation", "gemma-4-12b-it-f16",
                "gemma-head-0-8", "gemma-4-12b-it-f16|decode|gemma-head-0-8",
                now_us, now_us + 5_000_000, 1,
            )
            coordinator.admit(item, now_us)
            decisions.append(coordinator.dispatch(snapshot.route_id, now_us))
        self.assertTrue(all(
            isinstance(value, BatchDecision) and value.action == "WAIT"
            for value in decisions[:-1]
        ))
        self.assertIsInstance(decisions[-1], Launch)
        self.assertEqual(decisions[-1].reason, "target_batch_ready")
        self.assertEqual(decisions[-1].request_ids, request_ids)


if __name__ == "__main__":
    unittest.main()
