#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from power_frontier_policy import (  # noqa: E402
    BatchPoint,
    BoundaryCertificate,
    OperatingPlan,
    PolicyError,
    WorkItem,
    choose_batch,
    select_operating_plan,
    throughput_knee,
)


def work(
    request_id: str,
    *,
    key: str = "gemma:decode:k8",
    deadline_us: int = 10_000,
    priority: int = 1,
) -> WorkItem:
    return WorkItem(
        request_id=request_id,
        service_class="generation",
        model_id="gemma-4-12b",
        island_id="gemma-head-0-8",
        compatibility_key=key,
        arrival_us=0,
        deadline_us=deadline_us,
        priority_class=priority,
    )


MEMORY = (
    BatchPoint(1, 100),
    BatchPoint(2, 120),
    BatchPoint(4, 150),
)

MEMORY_B32 = (
    BatchPoint(1, 100),
    BatchPoint(8, 140),
    BatchPoint(16, 180),
    BatchPoint(32, 260),
)

COMPUTE = (
    BatchPoint(1, 100),
    BatchPoint(2, 105),
    BatchPoint(4, 220),
    BatchPoint(8, 430),
)


class PolicyTests(unittest.TestCase):
    def test_memory_bound_waits_for_target(self) -> None:
        decision = choose_batch(0, [work("r0"), work("r1")], MEMORY, "memory_bound")
        self.assertEqual(decision.action, "WAIT")
        self.assertEqual(decision.next_wake_us, 9850)

    def test_memory_bound_launches_largest_ready_batch(self) -> None:
        ready = [work(f"r{i}") for i in range(4)]
        decision = choose_batch(0, ready, MEMORY, "memory_bound")
        self.assertEqual((decision.action, decision.batch_size), ("LAUNCH", 4))
        self.assertEqual(decision.request_ids, ("r0", "r1", "r2", "r3"))

    def test_memory_bound_launches_certified_b32_target(self) -> None:
        ready = [work(f"r{i:02d}") for i in range(32)]
        decision = choose_batch(0, ready, MEMORY_B32, "memory_bound")
        self.assertEqual((decision.action, decision.batch_size), ("LAUNCH", 32))
        self.assertEqual(decision.reason, "target_batch_ready")

    def test_memory_bound_waits_for_b32_until_latest_safe_start(self) -> None:
        ready = [work(f"r{i:02d}") for i in range(31)]
        decision = choose_batch(0, ready, MEMORY_B32, "memory_bound")
        self.assertEqual(decision.action, "WAIT")
        self.assertEqual(decision.next_wake_us, 9740)

        released = choose_batch(9740, ready, MEMORY_B32, "memory_bound")
        self.assertEqual((released.action, released.batch_size), ("LAUNCH", 16))
        self.assertEqual(released.reason, "latest_start_reached")

    def test_memory_bound_uses_smaller_batch_when_b32_misses_slo(self) -> None:
        ready = [work(f"r{i:02d}", deadline_us=200) for i in range(32)]
        decision = choose_batch(0, ready, MEMORY_B32, "memory_bound")
        self.assertEqual((decision.action, decision.batch_size), ("LAUNCH", 16))
        self.assertEqual(decision.reason, "latest_start_reached")

    def test_latest_start_forces_current_batch(self) -> None:
        ready = [work("r0", deadline_us=130), work("r1", deadline_us=130)]
        decision = choose_batch(10, ready, MEMORY, "memory_bound")
        self.assertEqual((decision.action, decision.batch_size), ("LAUNCH", 2))
        self.assertEqual(decision.reason, "latest_start_reached")

    def test_high_priority_bypasses_wait(self) -> None:
        decision = choose_batch(0, [work("urgent", priority=0)], MEMORY, "memory_bound")
        self.assertEqual((decision.action, decision.batch_size), ("LAUNCH", 1))
        self.assertEqual(decision.reason, "urgent_bypass")

    def test_compute_bound_uses_smallest_batch_at_knee(self) -> None:
        self.assertEqual(throughput_knee(COMPUTE), 2)
        ready = [work(f"r{i}", key="bge:encode") for i in range(8)]
        decision = choose_batch(0, ready, COMPUTE, "compute_bound")
        self.assertEqual((decision.action, decision.batch_size), ("LAUNCH", 2))

    def test_incompatible_work_is_not_batched(self) -> None:
        ready = [work("a", key="model-a"), work("b", key="model-b")]
        decision = choose_batch(0, ready, MEMORY, "memory_bound")
        self.assertEqual(decision.action, "WAIT")
        self.assertEqual(decision.batch_size, 0)

    def test_compatibility_key_collision_rejected(self) -> None:
        first = work("a", key="same")
        second = WorkItem(
            request_id="b",
            service_class="embedding",
            model_id="bge",
            island_id="bge-encoder",
            compatibility_key="same",
            arrival_us=0,
            deadline_us=10_000,
            priority_class=1,
        )
        with self.assertRaisesRegex(PolicyError, "aliases incompatible work"):
            choose_batch(0, [first, second], MEMORY, "memory_bound")

    def test_priority_orders_launch_groups(self) -> None:
        ready = [
            work("low", key="low", priority=1),
            work("high", key="high", priority=0),
        ]
        decision = choose_batch(0, ready, MEMORY, "memory_bound")
        self.assertEqual(decision.request_ids, ("high",))

    def test_high_priority_wait_blocks_lower_priority_launch(self) -> None:
        ready = [
            work("high", key="high", priority=1),
            *[work(f"low-{i}", key="low", priority=2) for i in range(4)],
        ]
        decision = choose_batch(0, ready, MEMORY, "memory_bound")
        self.assertEqual(decision.action, "WAIT")
        self.assertEqual(decision.compatibility_key, "high")

    def test_high_priority_infeasible_blocks_lower_priority_launch(self) -> None:
        ready = [
            work("high", key="high", priority=1, deadline_us=50),
            *[work(f"low-{i}", key="low", priority=2) for i in range(4)],
        ]
        decision = choose_batch(0, ready, MEMORY, "memory_bound")
        self.assertEqual(decision.action, "NO_FEASIBLE")
        self.assertEqual(decision.compatibility_key, "high")

    def test_infeasible_deadline_fails_closed(self) -> None:
        decision = choose_batch(50, [work("r0", deadline_us=100)], MEMORY, "memory_bound")
        self.assertEqual(decision.action, "NO_FEASIBLE")

    def test_boundary_requires_every_gate(self) -> None:
        good = BoundaryCertificate("r0", True, True, True, True)
        bad = BoundaryCertificate("r0", True, True, True, False)
        self.assertTrue(good.admitted())
        self.assertFalse(bad.admitted())

    def test_operating_plan_selects_lowest_valid_energy(self) -> None:
        plans = [
            OperatingPlan("P0", "gpu0", 10, 10, 0, 1000, 0),
            OperatingPlan("P1", "gpu0", 10, 10, 1, 700, 0),
            OperatingPlan("P2", "gpu0", 10, 10, 0, 800, 0),
            OperatingPlan("P3", "gpu0", 10, 10, 0, 600, 0),
        ]
        self.assertEqual(select_operating_plan(plans, "gpu0", 10).label, "P3")

    def test_second_gpu_work_rejected(self) -> None:
        plans = [
            OperatingPlan(label, "gpu0", 10, 10, 0, 1000, int(label == "P2"))
            for label in ("P0", "P1", "P2", "P3")
        ]
        with self.assertRaisesRegex(PolicyError, "second GPU"):
            select_operating_plan(plans, "gpu0", 10)

    def test_gpu_identity_mismatch_rejected(self) -> None:
        plans = [
            OperatingPlan(label, "gpu1" if label == "P3" else "gpu0", 10, 10, 0, 1000, 0)
            for label in ("P0", "P1", "P2", "P3")
        ]
        with self.assertRaisesRegex(PolicyError, "UUID mismatch"):
            select_operating_plan(plans, "gpu0", 10)


if __name__ == "__main__":
    unittest.main()
