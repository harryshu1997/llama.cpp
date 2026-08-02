#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from priority_policy import (
    DispatchGroup,
    PriorityAdmissionController,
    PriorityPolicyError,
    PriorityRoute,
    PriorityWork,
    RejectDecision,
    RouteBatchPoint,
    WaitDecision,
)


DIGEST = "sha256:" + "1" * 64


def point(batch: int, duration: int, cuda: int) -> RouteBatchPoint:
    return RouteBatchPoint(batch, 1, 4, duration, cuda, (DIGEST,), "test")


def controller(
    r2_cuda_b4: int = 150, offload_enabled: bool = True,
) -> PriorityAdmissionController:
    routes = (
        PriorityRoute(
            "R0", ("cuda-prefix", "cuda-mid", "cuda-tail"),
            (point(1, 100, 100), point(4, 200, 400)), 5,
        ),
        PriorityRoute(
            "R2", ("op12-prefix", "op15-mid", "cuda-tail"),
            (point(1, 400, 60), point(4, 700, r2_cuda_b4)), 5,
        ),
    )
    capacities = {
        "cuda-prefix": 4,
        "cuda-mid": 4,
        "op12-prefix": 4,
        "op15-mid": 4,
        "cuda-tail": 8,
    }
    reserve = {
        "cuda-prefix": 4,
        "cuda-mid": 4,
        "op12-prefix": 0,
        "op15-mid": 0,
        "cuda-tail": 4,
    }
    return PriorityAdmissionController(
        routes, capacities, reserve, offload_enabled=offload_enabled,
    )


def work(
    request_id: int, priority: int, deadline: int = 10_000, arrival: int = 0,
) -> PriorityWork:
    return PriorityWork(request_id, arrival, deadline, priority, 1, 4)


class PriorityOrderTests(unittest.TestCase):
    def test_urgent_group_precedes_earlier_low_priority(self) -> None:
        policy = controller()
        for request_id in range(1, 5):
            policy.enqueue(work(request_id, 2), 0)
        for request_id in range(5, 9):
            policy.enqueue(work(request_id, 0), 0)
        decision = policy.decide(0)
        self.assertIsInstance(decision, DispatchGroup)
        self.assertEqual(decision.route_id, "R0")
        self.assertEqual(decision.request_ids, (5, 6, 7, 8))
        self.assertEqual(decision.batch_size, 4)
        self.assertEqual(decision.batch_wait_us, 5)
        self.assertEqual(decision.reason, "URGENT_PRIORITY_R0")

    def test_lone_urgent_request_launches_immediate_b1(self) -> None:
        policy = controller()
        policy.enqueue(work(1, 0), 0)
        decision = policy.decide(0)
        self.assertIsInstance(decision, DispatchGroup)
        self.assertEqual(decision.request_ids, (1,))
        self.assertEqual(decision.batch_size, 1)
        self.assertEqual(decision.batch_wait_us, 0)

    def test_priority_then_deadline_then_arrival_then_id(self) -> None:
        policy = controller()
        for item in (
            work(4, 2, 9000),
            work(3, 1, 9000),
            work(2, 1, 8000),
            work(1, 1, 8000),
        ):
            policy.enqueue(item, 0)
        self.assertEqual(
            [item.request_id for item in policy.pending()], [1, 2, 3, 4],
        )


class BatchAndReliefTests(unittest.TestCase):
    def test_low_priority_target_b4_requires_positive_relief(self) -> None:
        policy = controller()
        for request_id in range(1, 5):
            policy.enqueue(work(request_id, 2), 0)
        decision = policy.decide(0)
        self.assertIsInstance(decision, DispatchGroup)
        self.assertEqual(decision.route_id, "R2")
        self.assertEqual(decision.batch_size, 4)
        self.assertEqual(decision.predicted_cuda_relief_us, 250)
        self.assertEqual(decision.reason, "SERVER_RELIEVING_TARGET_BATCH")

    def test_no_relief_takes_immediate_server_fallback(self) -> None:
        policy = controller(r2_cuda_b4=450)
        for request_id in range(1, 5):
            policy.enqueue(work(request_id, 2), 0)
        decision = policy.decide(0)
        self.assertIsInstance(decision, DispatchGroup)
        self.assertEqual(decision.route_id, "R0")
        self.assertEqual(decision.request_ids, (1,))
        self.assertEqual(decision.batch_wait_us, 0)
        self.assertEqual(decision.reason, "OFFLOAD_HAS_NO_SERVER_RELIEF")

    def test_partial_batch_waits_only_to_latest_start(self) -> None:
        policy = controller()
        policy.enqueue(work(1, 2, deadline=1000), 0)
        decision = policy.decide(0)
        self.assertEqual(decision, WaitDecision(300, "BOUNDED_WAIT_FOR_TARGET_BATCH"))
        launched = policy.decide(300)
        self.assertIsInstance(launched, DispatchGroup)
        self.assertEqual(launched.route_id, "R2")
        self.assertEqual(launched.batch_size, 1)
        self.assertEqual(launched.reason, "LATEST_START_MEASURED_SMALL_BATCH")

    def test_unmeasured_batch_size_is_never_selected(self) -> None:
        policy = controller()
        for request_id in range(1, 4):
            policy.enqueue(work(request_id, 2, deadline=1000), 0)
        decision = policy.decide(300)
        self.assertIsInstance(decision, DispatchGroup)
        self.assertEqual(decision.batch_size, 1)

    def test_matched_control_keeps_low_priority_b4_on_cuda(self) -> None:
        policy = controller(offload_enabled=False)
        for request_id in range(1, 5):
            policy.enqueue(work(request_id, 2), 0)
        decision = policy.decide(0)
        self.assertIsInstance(decision, DispatchGroup)
        self.assertEqual(decision.route_id, "R0")
        self.assertEqual(decision.batch_size, 4)
        self.assertEqual(decision.batch_wait_us, 5)
        self.assertEqual(decision.predicted_cuda_relief_us, 0)
        self.assertEqual(decision.reason, "ALL_CUDA_MATCHED_CONTROL")

    def test_offload_flag_requires_boolean(self) -> None:
        with self.assertRaisesRegex(PriorityPolicyError, "must be boolean"):
            controller(offload_enabled=1)


class ReservationTests(unittest.TestCase):
    def test_low_b4_and_urgent_b4_share_tail_with_reserved_credits(self) -> None:
        policy = controller()
        for request_id in range(1, 5):
            policy.enqueue(work(request_id, 2), 0)
        low = policy.decide(0)
        self.assertIsInstance(low, DispatchGroup)
        for request_id in range(5, 9):
            policy.enqueue(work(request_id, 0), 0)
        urgent = policy.decide(0)
        self.assertIsInstance(urgent, DispatchGroup)
        self.assertEqual(urgent.route_id, "R0")
        self.assertEqual(urgent.batch_size, 4)
        state = policy.resource_state()
        self.assertEqual(state["active"]["cuda-tail"], 8)
        self.assertEqual(state["low_priority_active"]["cuda-tail"], 4)

    def test_second_low_group_waits_for_phone_credits(self) -> None:
        policy = controller()
        for request_id in range(1, 9):
            policy.enqueue(work(request_id, 2), 0)
        first = policy.decide(0)
        self.assertIsInstance(first, DispatchGroup)
        second = policy.decide(0)
        self.assertEqual(second.reason, "OFFLOAD_RESOURCE_BUSY")
        for request_id, epoch in zip(first.request_ids, first.route_epochs):
            policy.complete(request_id, epoch)
        second = policy.decide(1)
        self.assertIsInstance(second, DispatchGroup)
        self.assertEqual(second.request_ids, (5, 6, 7, 8))

    def test_stale_completion_does_not_release_credits(self) -> None:
        policy = controller()
        for request_id in range(1, 5):
            policy.enqueue(work(request_id, 2), 0)
        decision = policy.decide(0)
        with self.assertRaisesRegex(PriorityPolicyError, "epoch mismatch"):
            policy.complete(1, decision.route_epochs[0] + 1)
        self.assertEqual(policy.resource_state()["active"]["op12-prefix"], 4)

    def test_group_completion_is_atomic_on_stale_epoch(self) -> None:
        policy = controller()
        for request_id in range(1, 5):
            policy.enqueue(work(request_id, 2), 0)
        decision = policy.decide(0)
        epochs = list(decision.route_epochs)
        epochs[-1] += 1
        with self.assertRaisesRegex(PriorityPolicyError, "epoch mismatch"):
            policy.complete_group(decision.request_ids, epochs)
        self.assertEqual(set(policy.active()), {1, 2, 3, 4})
        self.assertEqual(policy.resource_state()["active"]["op12-prefix"], 4)


class FailureTests(unittest.TestCase):
    def test_impossible_deadline_rejects(self) -> None:
        policy = controller()
        policy.enqueue(work(1, 2, deadline=50), 0)
        decision = policy.decide(0)
        self.assertEqual(
            decision, RejectDecision(1, "NO_MEASURED_ROUTE_CAN_MEET_SLO"),
        )

    def test_duplicate_ownership_rejected(self) -> None:
        policy = controller()
        policy.enqueue(work(1, 1), 0)
        with self.assertRaisesRegex(PriorityPolicyError, "already owned"):
            policy.enqueue(work(1, 1), 0)

    def test_future_work_rejected(self) -> None:
        with self.assertRaisesRegex(PriorityPolicyError, "future request"):
            controller().enqueue(work(1, 1, arrival=1), 0)

    def test_bool_is_not_an_integer(self) -> None:
        with self.assertRaises(PriorityPolicyError):
            controller().enqueue(PriorityWork(True, 0, 1, 0, 1, 1), 0)

    def test_unprofiled_shape_is_rejected(self) -> None:
        policy = controller()
        policy.enqueue(PriorityWork(1, 0, 10_000, 0, 2, 4), 0)
        self.assertEqual(
            policy.decide(0),
            RejectDecision(1, "NO_MEASURED_ROUTE_CAN_MEET_SLO"),
        )


if __name__ == "__main__":
    unittest.main()
