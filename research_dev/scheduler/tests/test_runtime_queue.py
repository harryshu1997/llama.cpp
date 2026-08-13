#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import threading
import time
import unittest

from research_dev.scheduler import (
    BackgroundRuntimeMonitor,
    Decision,
    LeaseRecord,
    RuntimeDispatchQueue,
)


def decision(
    request_id: str,
    start_us: int,
    end_us: int,
    *,
    route_id: str = "phone-adreno",
    resource_id: str = "op15-adreno",
    lane: int = 0,
    token: str | None = None,
) -> Decision:
    lease = LeaseRecord(
        token=token or "lease-" + request_id,
        owner_id=request_id,
        lease_id="phone-full-task",
        resource_id=resource_id,
        lanes=(lane,),
        start_us=start_us,
        predicted_end_us=end_us - 10,
        reserved_until_us=end_us,
    )
    return Decision(
        request_id=request_id,
        workload_id="small-model",
        mode="enforce",
        route_id=route_id,
        granularity="task",
        start_us=start_us,
        finish_us=end_us - 10,
        finish_upper_us=end_us,
        service_us=end_us - start_us - 10,
        queue_us=start_us,
        queue_by_resource_us={"op15-adreno": start_us},
        blocking_resources=(() if start_us == 0 else ("op15-adreno",)),
        leases=(lease,),
        runtime_gate=None,
        energy_uj=10,
        energy_upper_uj=11,
        energy_breakdown=None,
        server_busy_us=end_us - start_us - 10,
        reason="VERIFIED_ENERGY_SAVING",
        rejected=(),
    )


class RuntimeDispatchQueueTests(unittest.TestCase):
    def test_same_route_uses_disjoint_cpu_lanes_concurrently(self) -> None:
        queue = RuntimeDispatchQueue()
        epoch_ns = time.monotonic_ns() - 1_000_000
        first = decision(
            "r0", 0, 20_000,
            route_id="desktop-cpu",
            resource_id="desktop-cpu",
            lane=0,
        )
        second = decision(
            "r1", 0, 20_000,
            route_id="desktop-cpu",
            resource_id="desktop-cpu",
            lane=1,
        )
        queue.admit(first, 0)
        queue.admit(second, 0)

        self.assertEqual(queue.wait("r0", epoch_ns).status, "ACQUIRED")
        self.assertEqual(queue.wait("r1", epoch_ns).status, "ACQUIRED")
        snapshot = queue.snapshot()
        self.assertEqual(
            snapshot["active_by_route"]["desktop-cpu"],
            ["r0", "r1"],
        )
        queue.complete("r0", 2_000)
        queue.complete("r1", 2_000)

    def test_different_routes_sharing_a_lane_do_not_overlap(self) -> None:
        queue = RuntimeDispatchQueue()
        epoch_ns = time.monotonic_ns() - 1_000_000
        first = decision(
            "r0", 0, 20_000,
            route_id="desktop-cpu",
            resource_id="desktop-cpu",
            lane=0,
        )
        second = decision(
            "r1", 0, 20_000,
            route_id="cpu-phone-ffn-split",
            resource_id="desktop-cpu",
            lane=0,
        )
        queue.admit(first, 0)
        queue.admit(second, 0)
        self.assertEqual(queue.wait("r0", epoch_ns).status, "ACQUIRED")

        receipt = []
        waiter = threading.Thread(
            target=lambda: receipt.append(queue.wait("r1", epoch_ns))
        )
        waiter.start()
        time.sleep(0.01)
        self.assertTrue(waiter.is_alive())
        queue.complete("r0", 2_000)
        waiter.join(1)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(receipt[0].wake_reason, "predecessor_completion")
        queue.complete("r1", 3_000)

    def test_busy_phone_request_waits_for_completion_event(self) -> None:
        queue = RuntimeDispatchQueue()
        epoch_ns = time.monotonic_ns() - 1_000_000
        first = decision("r0", 0, 2_000)
        second = decision("r1", 2_000, 4_000)
        queue.admit(first, 0)
        queue.admit(second, 0)
        self.assertEqual(queue.wait("r0", epoch_ns).status, "ACQUIRED")

        receipt = []
        waiter = threading.Thread(
            target=lambda: receipt.append(queue.wait("r1", epoch_ns))
        )
        waiter.start()
        time.sleep(0.01)
        self.assertTrue(waiter.is_alive())
        queue.complete("r0", 20_000)
        waiter.join(1)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(receipt[0].status, "ACQUIRED")
        self.assertEqual(receipt[0].wake_reason, "predecessor_completion")
        queue.complete("r1", 21_000)

    def test_overrun_conflict_wakes_request_for_replan(self) -> None:
        queue = RuntimeDispatchQueue()
        active = decision("r0", 0, 10_000)
        queued = decision("r1", 10_000, 20_000)
        queue.admit(active, 0)
        queue.admit(queued, 0)
        self.assertEqual(
            queue.conflicting_queued_requests(active, 15_000),
            ("r1",),
        )
        self.assertTrue(queue.require_replan("r1", "lease_overrun"))
        receipt = queue.wait("r1", time.monotonic_ns())
        self.assertEqual(receipt.status, "REPLAN_REQUIRED")
        self.assertEqual(receipt.wake_reason, "lease_overrun")
        queue.retire_replan("r1")

    def test_nonconflicting_route_is_not_replanned(self) -> None:
        queue = RuntimeDispatchQueue()
        active = decision("r0", 0, 10_000)
        cuda = replace(
            decision("r1", 0, 10_000, route_id="desktop-cuda"),
            leases=(LeaseRecord(
                token="lease-cuda",
                owner_id="r1",
                lease_id="cuda-full-task",
                resource_id="cuda0",
                lanes=(0,),
                start_us=0,
                predicted_end_us=9_990,
                reserved_until_us=10_000,
            ),),
        )
        queue.admit(active, 0)
        queue.admit(cuda, 0)
        self.assertEqual(
            queue.conflicting_queued_requests(active, 20_000),
            (),
        )


class BackgroundRuntimeMonitorTests(unittest.TestCase):
    def test_probe_runs_in_background_and_refreshes_on_notification(self) -> None:
        calls = []

        def probe() -> dict[str, object]:
            calls.append(time.monotonic_ns())
            return {"health": "healthy", "free_slots": 0}

        monitor = BackgroundRuntimeMonitor(
            {"phone": probe},
            refresh_interval_s=10,
            stale_after_s=20,
        )
        monitor.start()
        try:
            self.assertTrue(monitor.wait_until_populated(("phone",), 1))
            first_count = len(calls)
            monitor.request_refresh()
            deadline = time.monotonic() + 1
            while len(calls) == first_count and time.monotonic() < deadline:
                time.sleep(0.001)
            snapshot = monitor.snapshot("phone")
            self.assertGreater(len(calls), first_count)
            self.assertFalse(snapshot.stale)
            self.assertEqual(snapshot.value["free_slots"], 0)
        finally:
            monitor.stop()


if __name__ == "__main__":
    unittest.main()
