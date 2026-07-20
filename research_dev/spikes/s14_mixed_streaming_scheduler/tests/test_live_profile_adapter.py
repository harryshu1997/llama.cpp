#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
sys.path.insert(0, str(SPIKE))

from live_profile_adapter import ProfileAdapterError, load_op15_head_route  # noqa: E402
from power_frontier_policy import WorkItem  # noqa: E402
from priority_batch_runtime import Launch, PriorityBatchRuntime  # noqa: E402


ENERGY = SPIKE / "energy"


def b1_paths() -> list[Path]:
    return [ENERGY / f"stageb_op15_k8_b1_r{repeat}.json" for repeat in range(7)]


class LiveProfileAdapterTests(unittest.TestCase):
    def test_b1_result_binds_to_a_certified_route(self) -> None:
        route = load_op15_head_route(b1_paths(), 11)
        self.assertEqual([point.batch_size for point in route.points], [1])
        self.assertTrue(route.points[0].correctness_certificate_id.startswith("sha256:"))
        PriorityBatchRuntime((route,))

    def test_failed_larger_batches_are_rejected(self) -> None:
        for batch in (4, 8):
            with self.subTest(batch=batch), self.assertRaisesRegex(
                ProfileAdapterError, "completed passing sweep"
            ):
                load_op15_head_route([ENERGY / f"stageb_op15_k8_b{batch}_r0.json"], 11)

    def test_single_process_is_not_eligible(self) -> None:
        with self.assertRaisesRegex(ProfileAdapterError, "fewer than seven"):
            load_op15_head_route([ENERGY / "stageb_op15_k8_b1_r0.json"], 11)

    def test_runtime_cannot_infer_a_larger_batch(self) -> None:
        route = load_op15_head_route(b1_paths(), 11)
        runtime = PriorityBatchRuntime((route,))
        for index in range(4):
            runtime.enqueue(route.route_id, WorkItem(
                f"r{index}", "generation", "gemma-4-12b-it-f16", "gemma-head-0-8",
                "gemma|decode|c512", 0, 10_000_000, 1,
            ), 0)
        launch = runtime.decide(route.route_id, 0)
        self.assertIsInstance(launch, Launch)
        self.assertEqual(launch.batch_size, 1)


if __name__ == "__main__":
    unittest.main()
