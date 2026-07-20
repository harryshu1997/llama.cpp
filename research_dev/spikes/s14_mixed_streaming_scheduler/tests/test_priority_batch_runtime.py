#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from power_frontier_policy import (  # noqa: E402
    BatchDecision,
    BatchPoint,
    BoundaryCertificate,
    CertifiedBatchPoint,
    WorkItem,
)
from priority_batch_runtime import (  # noqa: E402
    BatchRuntimeError,
    Launch,
    PriorityBatchRuntime,
    RouteConfig,
)


def point(batch: int, duration: int) -> CertifiedBatchPoint:
    digest = f"sha256:{batch:064x}"
    return CertifiedBatchPoint(batch, duration, f"{digest}#correct", f"{digest}#placement")


MEMORY = (
    point(1, 100),
    point(2, 120),
    point(4, 150),
)
COMPUTE = (
    point(1, 100),
    point(2, 105),
    point(4, 220),
)


def route(route_id: str, roofline: str, points: tuple[BatchPoint, ...]) -> RouteConfig:
    if route_id == "server-bge":
        return RouteConfig(route_id, "embedding", "bge", "bge-encoder", "profile-bge", 7,
                           roofline, points)
    return RouteConfig(route_id, "generation", "gemma", "gemma-head", "profile-gemma", 9,
                       roofline, points)


def work(request_id: str, *, service: str = "generation", key: str = "gemma:c512",
         deadline: int = 10_000, priority: int = 1) -> WorkItem:
    if service == "embedding":
        return WorkItem(request_id, service, "bge", "bge-encoder", key, 0, deadline, priority)
    return WorkItem(request_id, service, "gemma", "gemma-head", key, 0, deadline, priority)


def cert(request_id: str, ok: bool = True) -> BoundaryCertificate:
    return BoundaryCertificate(request_id, ok, ok, ok, ok)


class RuntimeTests(unittest.TestCase):
    def test_route_rejects_uncertified_batch_point(self) -> None:
        bad = RouteConfig(
            "phone-gemma", "generation", "gemma", "gemma-head", "profile-gemma",
            9, "memory_bound", (BatchPoint(1, 100),),
        )
        with self.assertRaisesRegex(BatchRuntimeError, "uncertified batch point"):
            PriorityBatchRuntime((bad,))

    def test_route_rejects_unbound_certificate_id(self) -> None:
        bad_point = CertifiedBatchPoint(1, 100, "correct", "placement")
        bad = RouteConfig(
            "phone-gemma", "generation", "gemma", "gemma-head", "profile-gemma",
            9, "memory_bound", (bad_point,),
        )
        with self.assertRaisesRegex(BatchRuntimeError, "sha256"):
            PriorityBatchRuntime((bad,))

    def test_memory_batch_waits_then_launches_at_latest_start(self) -> None:
        runtime = PriorityBatchRuntime((route("phone-gemma", "memory_bound", MEMORY),))
        runtime.enqueue("phone-gemma", work("r0", deadline=200), 0)
        runtime.enqueue("phone-gemma", work("r1", deadline=200), 0)
        decision = runtime.decide("phone-gemma", 0)
        self.assertIsInstance(decision, BatchDecision)
        self.assertEqual((decision.action, decision.next_wake_us), ("WAIT", 50))
        launch = runtime.decide("phone-gemma", 50)
        self.assertIsInstance(launch, Launch)
        self.assertEqual((launch.batch_size, launch.request_ids), (2, ("r0", "r1")))

    def test_compute_route_launches_measured_knee(self) -> None:
        runtime = PriorityBatchRuntime((route("server-bge", "compute_bound", COMPUTE),))
        for index in range(4):
            runtime.enqueue("server-bge", work(f"e{index}", service="embedding", key="bge:l32"), 0)
        launch = runtime.decide("server-bge", 0)
        self.assertIsInstance(launch, Launch)
        self.assertEqual(launch.batch_size, 2)

    def test_independent_server_and_phone_routes_launch_concurrently(self) -> None:
        runtime = PriorityBatchRuntime((
            route("server-bge", "compute_bound", COMPUTE),
            route("phone-gemma", "memory_bound", MEMORY),
        ))
        for index in range(4):
            runtime.enqueue("server-bge", work(f"e{index}", service="embedding", key="bge:l32"), 0)
            runtime.enqueue("phone-gemma", work(f"g{index}"), 0)
        server = runtime.decide("server-bge", 0)
        phone = runtime.decide("phone-gemma", 0)
        self.assertIsInstance(server, Launch)
        self.assertIsInstance(phone, Launch)
        self.assertEqual((server.batch_size, phone.batch_size), (2, 4))

    def test_priority_wait_blocks_lower_priority_launch(self) -> None:
        runtime = PriorityBatchRuntime((route("phone-gemma", "memory_bound", MEMORY),))
        runtime.enqueue("phone-gemma", work("high", key="high", priority=1), 0)
        for index in range(4):
            runtime.enqueue("phone-gemma", work(f"low{index}", key="low", priority=2), 0)
        decision = runtime.decide("phone-gemma", 0)
        self.assertIsInstance(decision, BatchDecision)
        self.assertEqual((decision.action, decision.compatibility_key), ("WAIT", "high"))

    def test_route_signature_mismatch_rejected(self) -> None:
        runtime = PriorityBatchRuntime((route("phone-gemma", "memory_bound", MEMORY),))
        with self.assertRaisesRegex(BatchRuntimeError, "route signature"):
            runtime.enqueue("phone-gemma", work("e0", service="embedding"), 0)

    def test_wrong_runtime_types_rejected(self) -> None:
        with self.assertRaisesRegex(BatchRuntimeError, "RouteConfig"):
            PriorityBatchRuntime(("not-a-route",))
        runtime = PriorityBatchRuntime((route("phone-gemma", "memory_bound", MEMORY),))
        with self.assertRaisesRegex(BatchRuntimeError, "WorkItem"):
            runtime.enqueue("phone-gemma", {"request_id": "r0"}, 0)

    def test_duplicate_and_future_work_rejected(self) -> None:
        runtime = PriorityBatchRuntime((route("phone-gemma", "memory_bound", MEMORY),))
        item = work("r0")
        runtime.enqueue("phone-gemma", item, 0)
        with self.assertRaisesRegex(BatchRuntimeError, "duplicate request_id"):
            runtime.enqueue("phone-gemma", item, 0)
        future = WorkItem("future", "generation", "gemma", "gemma-head", "gemma:c512", 10, 20, 1)
        with self.assertRaisesRegex(BatchRuntimeError, "future work"):
            runtime.enqueue("phone-gemma", future, 0)

    def test_exact_certificates_complete_launch(self) -> None:
        runtime = PriorityBatchRuntime((route("phone-gemma", "memory_bound", MEMORY),))
        for index in range(4):
            runtime.enqueue("phone-gemma", work(f"r{index}"), 0)
        launch = runtime.decide("phone-gemma", 0)
        self.assertIsInstance(launch, Launch)
        done = runtime.complete("phone-gemma", launch.launch_id, launch.route_epoch, 150,
                                [cert(request_id) for request_id in launch.request_ids])
        self.assertEqual({item.status for item in done}, {"completed"})
        self.assertEqual(runtime.snapshot()["inflight"], ())

    def test_missing_duplicate_and_stale_certificates_rejected(self) -> None:
        runtime = PriorityBatchRuntime((route("phone-gemma", "memory_bound", MEMORY),))
        for index in range(4):
            runtime.enqueue("phone-gemma", work(f"r{index}"), 0)
        launch = runtime.decide("phone-gemma", 0)
        self.assertIsInstance(launch, Launch)
        with self.assertRaisesRegex(BatchRuntimeError, "certificate set"):
            runtime.complete("phone-gemma", launch.launch_id, launch.route_epoch, 150,
                             [cert(launch.request_ids[0])])
        with self.assertRaisesRegex(BatchRuntimeError, "duplicate certificate"):
            runtime.complete("phone-gemma", launch.launch_id, launch.route_epoch, 150,
                             [cert(x) for x in launch.request_ids] + [cert(launch.request_ids[0])])
        with self.assertRaisesRegex(BatchRuntimeError, "route_epoch"):
            runtime.complete("phone-gemma", launch.launch_id, launch.route_epoch + 1, 150,
                             [cert(x) for x in launch.request_ids])

    def test_failed_boundary_requires_server_fallback(self) -> None:
        runtime = PriorityBatchRuntime((route(
            "phone-gemma", "memory_bound", (point(1, 100),)
        ),))
        runtime.enqueue("phone-gemma", work("r0"), 0)
        launch = runtime.decide("phone-gemma", 0)
        self.assertIsInstance(launch, Launch)
        done = runtime.complete("phone-gemma", launch.launch_id, launch.route_epoch, 100, [cert("r0", False)])
        self.assertEqual(done[0].status, "fallback_required")
        self.assertEqual(runtime.fallback_request_ids(), ("r0",))

    def test_late_valid_result_is_not_admitted(self) -> None:
        runtime = PriorityBatchRuntime((route(
            "phone-gemma", "memory_bound", (point(1, 100),)
        ),))
        runtime.enqueue("phone-gemma", work("r0", deadline=100), 0)
        launch = runtime.decide("phone-gemma", 0)
        self.assertIsInstance(launch, Launch)
        done = runtime.complete("phone-gemma", launch.launch_id, launch.route_epoch, 101, [cert("r0")])
        self.assertEqual(done[0].status, "tardy_result")

    def test_busy_route_and_foreign_launch_rejected(self) -> None:
        runtime = PriorityBatchRuntime((route(
            "phone-gemma", "memory_bound", (point(1, 100),)
        ),))
        runtime.enqueue("phone-gemma", work("r0"), 0)
        launch = runtime.decide("phone-gemma", 0)
        self.assertIsInstance(launch, Launch)
        with self.assertRaisesRegex(BatchRuntimeError, "in-flight launch"):
            runtime.decide("phone-gemma", 1)
        with self.assertRaisesRegex(BatchRuntimeError, "launch_id"):
            runtime.fail_launch("phone-gemma", launch.launch_id + 1, launch.route_epoch, 1)

    def test_malformed_certificate_is_atomic(self) -> None:
        runtime = PriorityBatchRuntime((route("phone-gemma", "memory_bound", MEMORY),))
        for index in range(4):
            runtime.enqueue("phone-gemma", work(f"r{index}"), 0)
        launch = runtime.decide("phone-gemma", 0)
        self.assertIsInstance(launch, Launch)
        certificates = [cert(request_id) for request_id in launch.request_ids]
        certificates[-1] = BoundaryCertificate(launch.request_ids[-1], 1, True, True, True)
        with self.assertRaisesRegex(BatchRuntimeError, "identity_ok must be bool"):
            runtime.complete("phone-gemma", launch.launch_id, launch.route_epoch, 150, certificates)
        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["completed"], ())
        self.assertEqual(snapshot["inflight"][0][1], launch.launch_id)

    def test_non_certificate_is_rejected_atomically(self) -> None:
        runtime = PriorityBatchRuntime((route(
            "phone-gemma", "memory_bound", (point(1, 100),)
        ),))
        runtime.enqueue("phone-gemma", work("r0"), 0)
        launch = runtime.decide("phone-gemma", 0)
        self.assertIsInstance(launch, Launch)
        with self.assertRaisesRegex(BatchRuntimeError, "BoundaryCertificate"):
            runtime.complete("phone-gemma", launch.launch_id, launch.route_epoch, 100, [object()])
        self.assertNotEqual(runtime.snapshot()["inflight"], ())

    def test_failed_launch_releases_lane_and_preserves_accounting(self) -> None:
        runtime = PriorityBatchRuntime((route(
            "phone-gemma", "memory_bound", (point(1, 100),)
        ),))
        runtime.enqueue("phone-gemma", work("r0"), 0)
        launch = runtime.decide("phone-gemma", 0)
        self.assertIsInstance(launch, Launch)
        done = runtime.fail_launch("phone-gemma", launch.launch_id, launch.route_epoch, 10)
        self.assertEqual(done[0].status, "fallback_required")
        self.assertEqual(runtime.snapshot()["inflight"], ())


if __name__ == "__main__":
    unittest.main()
