#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import threading
import time
from types import MappingProxyType
import unittest

from research_dev.scheduler import (
    BackgroundRuntimeMonitor,
    Decision,
    LeaseRecord,
    RuntimeDispatchQueue,
)
from research_dev.scheduler._internal.runtime_decode_cohort import RuntimeDecodeCohortBinding


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


def decision_with_lanes(
    request_id: str,
    start_us: int,
    end_us: int,
    lanes_by_resource: dict[str, tuple[int, ...]],
) -> Decision:
    base = decision(request_id, start_us, end_us)
    leases = tuple(
        replace(
            base.leases[0],
            token=f"lease-{request_id}-{index}",
            lease_id=f"lease-{resource_id}",
            resource_id=resource_id,
            lanes=lanes,
        )
        for index, (resource_id, lanes) in enumerate(
            sorted(lanes_by_resource.items())
        )
    )
    return replace(base, leases=leases)


class RuntimeDispatchQueueTests(unittest.TestCase):
    def test_decode_cohort_members_dispatch_concurrently(self):
        queue = RuntimeDispatchQueue()
        members = ("cohort-a", "cohort-b")
        first = decision(members[0], 0, 10_000_000, token="shared-execution")
        execution = replace(first.leases[0], owner_id="cohort")
        preparation = replace(execution, token="shared-preparation", lease_id="prepare")
        binding = RuntimeDecodeCohortBinding(
            "cohort", "key", members[0], members, "policy",
            (preparation.token, execution.token), 2, 4, True,
        )
        for member in members:
            queue.admit(replace(first, request_id=member, leases=(preparation, execution)),
                        0, residency_transition_barrier=True, decode_cohort=binding)
        start = threading.Barrier(3)
        acquired = []
        errors = []

        def dispatch(member):
            try:
                start.wait(timeout=1)
                acquired.append(queue.wait(member, time.monotonic_ns()))
            except Exception as error:
                errors.append(error)

        workers = [threading.Thread(target=dispatch, args=(member,), daemon=True)
                   for member in members]
        for worker in workers:
            worker.start()
        start.wait(timeout=1)
        try:
            for worker in workers:
                worker.join(1)
            self.assertFalse(any(worker.is_alive() for worker in workers))
            self.assertEqual(errors, [])
            self.assertEqual({row.request_id for row in acquired}, set(members))
            self.assertTrue(all(row.status == "ACQUIRED" for row in acquired))
            self.assertEqual(set(queue.snapshot()["active"]), set(members))
            queue.release_prepare_leases(members[0], (preparation.token,))
            self.assertFalse(queue._active_conflict(
                queue._entries[members[1]], excluding_request_id=members[1]))
            queue.restore(queue.checkpoint())
            self.assertEqual(queue._entries[members[0]].decode_cohort, binding)
            self.assertFalse(queue._active_conflict(
                queue._entries[members[1]], excluding_request_id=members[1]))
        finally:
            for member in members:
                state = queue.snapshot()["entry_states"].get(member, {}).get("state")
                if state == "ACTIVE":
                    queue.complete(member, 100)
                elif state == "QUEUED":
                    queue.cancel_queued(member, 100)
            for worker in workers:
                worker.join(1)

    def test_fifth_request_waits_until_last_cohort_member_releases(self):
        queue = RuntimeDispatchQueue()
        members = tuple(f"member-{index}" for index in range(4))
        first = decision(members[0], 0, 10_000_000, token="cohort-reservation")
        first = replace(first, leases=(replace(first.leases[0], owner_id="cohort"),))
        binding = RuntimeDecodeCohortBinding(
            "cohort", "key", members[0], members, "policy",
            (first.leases[0].token,), 4, 4, True,
        )
        for member in members:
            queue.admit(replace(first, request_id=member), 0, decode_cohort=binding)
            self.assertEqual(queue.wait(member, time.monotonic_ns()).status, "ACQUIRED")
        queue.admit(decision("fifth", 0, 10_000_000), 1)
        self.assertFalse(queue._causal_ready(queue._entries["fifth"]))
        self.assertTrue(queue._active_conflict(queue._entries["fifth"]))
        acquired = threading.Event()
        errors = []

        def dispatch_fifth():
            try:
                receipt = queue.wait("fifth", time.monotonic_ns())
                if receipt.status == "ACQUIRED":
                    acquired.set()
            except Exception as error:
                errors.append(error)

        waiter = threading.Thread(target=dispatch_fifth, daemon=True)
        waiter.start()
        try:
            self.assertFalse(acquired.wait(0.02))
            for member in members[:-1]:
                queue.complete(member, 100)
                self.assertFalse(acquired.wait(0.02))
                self.assertTrue(queue._active_conflict(queue._entries["fifth"]))
            queue.complete(members[-1], 100)
            self.assertTrue(acquired.wait(1))
            self.assertEqual(errors, [])
            self.assertEqual(set(queue.snapshot()["active"]), {"fifth"})
        finally:
            if queue.snapshot()["entry_states"].get("fifth", {}).get("state") == "ACTIVE":
                queue.complete("fifth", 200)
            else:
                queue.cancel_queued("fifth", 200)
            waiter.join(1)

    def test_waiter_sleeps_while_another_thread_replans_request(
        self,
    ) -> None:
        queue = RuntimeDispatchQueue()
        original = decision("replanned", 0, 10_000)
        queue.admit(original, 0)
        self.assertTrue(queue.require_replan(
            original.request_id, "residency_projection_invalid"
        ))
        wake = queue.wait_ready(
            original.request_id, time.monotonic_ns()
        )
        self.assertEqual(wake.status, "REPLAN_REQUIRED")
        queue.retire_replan(
            original.request_id, wake.queue_generation
        )

        receipts = []
        errors = []

        def wait_for_replacement() -> None:
            try:
                receipts.append(queue.wait_ready(
                    original.request_id, time.monotonic_ns()
                ))
            except BaseException as exc:
                errors.append(exc)

        waiter = threading.Thread(
            target=wait_for_replacement,
            daemon=True,
        )
        waiter.start()
        waiter.join(0.05)
        self.assertTrue(waiter.is_alive())

        replacement = decision(original.request_id, 0, 20_000)
        queue.admit(replacement, 1)
        waiter.join(0.1)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0].status, "ACQUIRED")
        self.assertEqual(receipts[0].route_id, replacement.route_id)

    def test_capacity_frontier_excludes_nonrunnable_lane_follower(
        self,
    ) -> None:
        queue = RuntimeDispatchQueue()
        owner = decision_with_lanes(
            "owner",
            0,
            10_000_000,
            {"coordinator": (0, 1), "cuda": (0, 1, 2)},
        )
        first = decision_with_lanes(
            "first",
            10_000_000,
            20_000_000,
            {"coordinator": (0,), "cuda": (0,)},
        )
        second = decision_with_lanes(
            "second",
            10_000_000,
            20_000_000,
            {"coordinator": (1,), "cuda": (1,)},
        )
        later = decision_with_lanes(
            "later",
            20_000_000,
            30_000_000,
            {"coordinator": (0,), "cuda": (2,)},
        )
        queue.admit(owner, 0)
        queue.admit(first, 1)
        queue.admit(second, 2)
        queue.admit(later, 3)
        self.assertEqual(
            queue.wait(owner.request_id, time.monotonic_ns()).status,
            "ACQUIRED",
        )

        self.assertEqual(
            queue.causal_follower_frontier(
                owner.request_id,
                (first.request_id, second.request_id, later.request_id),
            ),
            (first.request_id, second.request_id),
        )

        replan_queue = RuntimeDispatchQueue()
        old_first = decision_with_lanes(
            "replanned",
            10_000_000,
            20_000_000,
            {"coordinator": (0,), "cuda": (0, 2)},
        )
        blocked = decision_with_lanes(
            "blocked",
            20_000_000,
            30_000_000,
            {"coordinator": (0,), "cuda": (2,)},
        )
        replan_queue.admit(old_first, 0)
        replan_queue.admit(blocked, 1)
        self.assertTrue(replan_queue.require_replan(
            old_first.request_id, "capacity_released_early"
        ))
        receipt = replan_queue.wait(
            old_first.request_id, time.monotonic_ns()
        )
        self.assertEqual(receipt.status, "REPLAN_REQUIRED")
        replan_queue.retire_replan(old_first.request_id)
        new_first = decision_with_lanes(
            old_first.request_id,
            10_000_000,
            20_000_000,
            {"coordinator": (0,), "cuda": (0,)},
        )
        replan_queue.admit(new_first, 2)
        self.assertEqual(
            replan_queue.replanned_capacity_frontier(
                old_first.request_id, old_first
            ),
            (),
        )

    def test_hot_replan_exposes_only_next_freed_lane_successor(self) -> None:
        queue = RuntimeDispatchQueue()
        owner = replace(
            decision("cold-owner", 0, 10_000_000),
            leases=(replace(
                decision("cold-owner", 0, 10_000_000).leases[0],
                lanes=(0, 1),
            ),),
        )
        first = replace(
            decision("cold-first", 10_000_000, 20_000_000),
            leases=(replace(
                decision("cold-first", 10_000_000, 20_000_000).leases[0],
                lanes=(0, 1),
            ),),
        )
        second = replace(
            decision("cold-second", 20_000_000, 30_000_000),
            leases=(replace(
                decision("cold-second", 20_000_000, 30_000_000).leases[0],
                lanes=(0, 1),
            ),),
        )
        queue.admit(owner, 0)
        queue.admit(first, 1)
        queue.admit(second, 2)
        epoch_ns = time.monotonic_ns()
        self.assertEqual(queue.wait(owner.request_id, epoch_ns).status, "ACQUIRED")
        self.assertTrue(queue.require_replan(
            first.request_id,
            "capacity_released_early",
            defer_behind_predecessors=True,
        ))
        queue.complete(
            owner.request_id,
            50_000,
            early_replan_request_ids=(first.request_id,),
        )
        self.assertEqual(
            queue.wait(first.request_id, epoch_ns).status,
            "REPLAN_REQUIRED",
        )

        queue.retire_replan(first.request_id)
        warm_first = decision(first.request_id, 50_000, 1_000_000)
        queue.admit(warm_first, 50_000)
        frontier = queue.replanned_capacity_frontier(
            first.request_id, first
        )
        self.assertEqual(frontier, (second.request_id,))
        self.assertTrue(queue.require_replan(
            second.request_id,
            "capacity_released_early",
            defer_behind_predecessors=False,
        ))
        self.assertEqual(
            queue.wait(second.request_id, epoch_ns).status,
            "REPLAN_REQUIRED",
        )

        queue.retire_replan(second.request_id)
        warm_second = decision(
            second.request_id, 50_000, 1_000_000, lane=1
        )
        queue.admit(warm_second, 50_000)
        first_active = queue.wait(first.request_id, epoch_ns)
        second_active = queue.wait(second.request_id, epoch_ns)

        self.assertEqual(first_active.status, "ACQUIRED")
        self.assertEqual(second_active.status, "ACQUIRED")
        self.assertEqual(
            set(queue.snapshot()["active"]),
            {first.request_id, second.request_id},
        )

    def test_early_completion_wakes_only_first_lane_frontier(self) -> None:
        queue = RuntimeDispatchQueue()
        owner = decision("owner", 0, 10_000_000)
        owner = replace(
            owner,
            leases=(replace(owner.leases[0], lanes=(0, 1)),),
        )
        first = decision("first", 10_000_000, 20_000_000)
        other_lane = decision(
            "other-lane",
            10_000_000,
            20_000_000,
            lane=1,
        )
        later = decision("later", 20_000_000, 30_000_000)
        queue.admit(owner, 0)
        queue.admit(first, 1)
        queue.admit(other_lane, 2)
        queue.admit(later, 3)
        epoch_ns = time.monotonic_ns()
        self.assertEqual(
            queue.wait(owner.request_id, epoch_ns).status, "ACQUIRED"
        )
        first_receipts = []
        other_lane_receipts = []
        first_waiter = threading.Thread(
            target=lambda: first_receipts.append(
                queue.wait_ready(first.request_id, epoch_ns)
            ),
            daemon=True,
        )
        other_lane_waiter = threading.Thread(
            target=lambda: other_lane_receipts.append(
                queue.wait_ready(other_lane.request_id, epoch_ns)
            ),
            daemon=True,
        )
        first_waiter.start()
        other_lane_waiter.start()
        time.sleep(0.01)
        frontier = queue.early_completion_frontier(
            owner.request_id,
            50_000,
            {owner.leases[0].token: owner.leases[0].reserved_until_us},
        )
        self.assertEqual(
            frontier, (first.request_id, other_lane.request_id)
        )
        for request_id in frontier:
            self.assertTrue(queue.require_replan(
                request_id,
                "capacity_released_early",
                defer_behind_predecessors=True,
            ))

        started = time.monotonic()
        queue.complete(
            owner.request_id,
            50_000,
            early_replan_request_ids=frontier,
        )
        first_waiter.join(0.1)
        other_lane_waiter.join(0.1)

        self.assertFalse(first_waiter.is_alive())
        self.assertFalse(other_lane_waiter.is_alive())
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertEqual(first_receipts[0].status, "REPLAN_REQUIRED")
        self.assertEqual(
            other_lane_receipts[0].status, "REPLAN_REQUIRED"
        )
        self.assertEqual(
            first_receipts[0].wake_reason,
            "capacity_released_early",
        )
        self.assertEqual(
            queue.replan_required_requests(),
            (first.request_id, other_lane.request_id),
        )
        self.assertEqual(
            queue.snapshot()["causal_predecessors"],
            {later.request_id: [first.request_id]},
        )

    def test_early_completion_keeps_later_selected_follower_deferred(
        self,
    ) -> None:
        queue = RuntimeDispatchQueue()
        owner = decision_with_lanes(
            "owner",
            0,
            10_000_000,
            {"gpu": (0, 1)},
        )
        first = decision(
            "first",
            10_000_000,
            20_000_000,
            resource_id="gpu",
        )
        later = decision(
            "later",
            20_000_000,
            30_000_000,
            resource_id="gpu",
        )
        queue.admit(owner, 0)
        queue.admit(first, 1)
        queue.admit(later, 2)
        self.assertEqual(
            queue.wait(owner.request_id, time.monotonic_ns()).status,
            "ACQUIRED",
        )
        frontier = queue.early_completion_frontier(
            owner.request_id,
            50_000,
            {
                lease.token: lease.reserved_until_us
                for lease in owner.leases
            },
            {"gpu": 2},
        )
        self.assertEqual(frontier, (first.request_id, later.request_id))
        for request_id in frontier:
            self.assertTrue(queue.require_replan(
                request_id,
                "capacity_released_early",
                defer_behind_predecessors=True,
            ))

        queue.release_capacity(
            owner.request_id,
            50_000,
            early_replan_request_ids=frontier,
        )

        states = queue.snapshot()["entry_states"]
        self.assertEqual(states[first.request_id]["state"], "REPLAN_REQUIRED")
        self.assertEqual(states[later.request_id]["state"], "DEFERRED_REPLAN")
        self.assertEqual(
            queue.replan_required_requests(),
            (first.request_id,),
        )

    def test_early_completion_replans_cold_follower_for_free_hot_lane(
        self,
    ) -> None:
        queue = RuntimeDispatchQueue()
        owner = decision_with_lanes(
            "owner",
            0,
            10_000_000,
            {"coordinator": (0,), "cuda": (0,)},
        )
        peer = decision_with_lanes(
            "peer",
            10_000_000,
            20_000_000,
            {"coordinator": (0,), "cuda": (0,)},
        )
        follower = decision_with_lanes(
            "follower",
            20_000_000,
            30_000_000,
            {"coordinator": (0, 1), "cuda": (0, 1)},
        )
        queue.admit(owner, 0)
        queue.admit(peer, 1)
        queue.admit(follower, 2)
        epoch_ns = time.monotonic_ns()
        self.assertEqual(queue.wait("owner", epoch_ns).status, "ACQUIRED")
        self.assertTrue(queue.require_replan(
            "peer",
            "capacity_released_early",
            defer_behind_predecessors=False,
        ))
        self.assertEqual(
            queue.wait("peer", epoch_ns).status, "REPLAN_REQUIRED"
        )
        queue.retire_replan("peer")
        queue.admit(decision_with_lanes(
            "peer",
            1,
            20_000_000,
            {"coordinator": (1,), "cuda": (1,)},
        ), 1)
        self.assertEqual(queue.wait("peer", epoch_ns).status, "ACQUIRED")
        self.assertEqual(
            queue.snapshot()["causal_predecessors"],
            {"follower": ["owner", "peer"]},
        )

        frontier = queue.early_completion_frontier(
            "owner",
            50_000,
            {
                lease.token: lease.reserved_until_us
                for lease in owner.leases
            },
            {"coordinator": 2, "cuda": 2},
        )

        self.assertEqual(frontier, ("follower",))

    def test_rolling_lease_wakes_overdue_successor_as_early_capacity(
        self,
    ) -> None:
        queue = RuntimeDispatchQueue()
        owner = decision("rolling-owner", 0, 40_000)
        follower = decision("rolling-follower", 40_000, 140_000)
        queue.admit(owner, 0)
        queue.admit(follower, 1)
        epoch_ns = time.monotonic_ns()
        self.assertEqual(
            queue.wait(owner.request_id, epoch_ns).status, "ACQUIRED"
        )
        self.assertTrue(queue.require_replan(
            follower.request_id,
            "lease_upper_bound_overrun",
            defer_behind_predecessors=True,
        ))

        frontier = queue.early_completion_frontier(
            owner.request_id,
            50_000,
            {owner.leases[0].token: 100_000},
        )

        self.assertEqual(frontier, (follower.request_id,))
        self.assertTrue(queue.require_replan(
            follower.request_id,
            "capacity_released_early",
            defer_behind_predecessors=True,
        ))
        started = time.monotonic()
        queue.complete(
            owner.request_id,
            50_000,
            early_replan_request_ids=frontier,
        )
        receipt = queue.wait(follower.request_id, epoch_ns)
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertEqual(receipt.status, "REPLAN_REQUIRED")
        self.assertEqual(receipt.wake_reason, "capacity_released_early")

    def test_physical_capacity_release_precedes_terminal_measurement(
        self,
    ) -> None:
        queue = RuntimeDispatchQueue()
        owner = decision("measured-owner", 0, 10_000_000)
        follower = decision(
            "measured-follower", 10_000_000, 20_000_000
        )
        queue.admit(owner, 0)
        queue.admit(follower, 1)
        epoch_ns = time.monotonic_ns()
        self.assertEqual(
            queue.wait(owner.request_id, epoch_ns).status, "ACQUIRED"
        )
        self.assertTrue(queue.require_replan(
            follower.request_id,
            "capacity_released_early",
            defer_behind_predecessors=True,
        ))
        follower_receipts = []
        waiter = threading.Thread(
            target=lambda: follower_receipts.append(
                queue.wait_ready(follower.request_id, epoch_ns)
            ),
            daemon=True,
        )
        waiter.start()
        time.sleep(0.01)

        started = time.monotonic()
        queue.release_capacity(
            owner.request_id,
            50_000,
            early_replan_request_ids=(follower.request_id,),
        )
        waiter.join(0.1)

        self.assertFalse(waiter.is_alive())
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertEqual(
            follower_receipts[0].status, "REPLAN_REQUIRED"
        )
        self.assertEqual(
            follower_receipts[0].wake_reason,
            "capacity_released_early",
        )
        self.assertEqual(
            queue.snapshot()["entry_states"][owner.request_id]["state"],
            "FINISHING",
        )

        queue.complete(owner.request_id, 50_000)
        self.assertNotIn(
            owner.request_id, queue.snapshot()["entry_states"]
        )

    def test_expired_lease_requires_replan_before_acquire(self) -> None:
        queue = RuntimeDispatchQueue()
        owner = decision("owner", 0, 10_000)
        queued = decision("expired", 10_000, 20_000)
        queue.admit(owner, 0)
        queue.admit(queued, 0)
        epoch_ns = time.monotonic_ns() - 100_000_000
        self.assertEqual(
            queue.wait(owner.request_id, epoch_ns).status, "ACQUIRED"
        )
        queue.complete(owner.request_id, 100_000)

        receipt = queue.wait(
            queued.request_id,
            epoch_ns,
        )

        self.assertEqual(receipt.status, "REPLAN_REQUIRED")
        self.assertEqual(
            receipt.wake_reason,
            "lease_coverage_expired_before_dispatch",
        )
        self.assertEqual(queue.snapshot()["active"], {})
        self.assertEqual(
            queue.replan_required_requests(), (queued.request_id,)
        )

    def test_replan_frontier_is_not_blocked_by_earlier_arrival_dependent(
        self,
    ) -> None:
        queue = RuntimeDispatchQueue()
        dependent = decision("earlier-arrival", 20_000, 30_000)
        frontier = decision("causal-frontier", 0, 10_000)
        queue.admit(dependent, 0)
        queue.admit(frontier, 0)
        self.assertEqual(
            queue.snapshot()["causal_predecessors"],
            {"earlier-arrival": ["causal-frontier"]},
        )
        self.assertTrue(queue.require_replan(
            dependent.request_id, "residency_observation_changed"
        ))
        self.assertTrue(queue.require_replan(
            frontier.request_id, "residency_observation_changed"
        ))

        receipts = []
        waiter = threading.Thread(
            target=lambda: receipts.append(
                queue.wait_ready(frontier.request_id, time.monotonic_ns())
            ),
            daemon=True,
        )
        waiter.start()
        waiter.join(0.2)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(receipts[0].status, "REPLAN_REQUIRED")

    def test_residency_transition_preserves_conflicting_arrival_order(
        self,
    ) -> None:
        queue = RuntimeDispatchQueue()
        transition = decision("earlier-transition", 20_000, 30_000)
        later_hot = decision("later-hot", 0, 10_000)

        queue.admit(
            transition,
            0,
            residency_transition_barrier=True,
        )
        queue.admit(later_hot, 1)

        self.assertEqual(
            queue.snapshot()["causal_predecessors"],
            {"later-hot": ["earlier-transition"]},
        )

    def test_replan_keeps_existing_causal_order_acyclic(self) -> None:
        queue = RuntimeDispatchQueue()
        later = decision("later", 30_000, 40_000)
        earlier = decision("earlier", 10_000, 20_000)
        middle = decision("middle", 20_000, 30_000)
        queue.admit(later, 0)
        queue.admit(earlier, 1)
        queue.admit(middle, 2)

        self.assertEqual(
            queue.snapshot()["causal_predecessors"],
            {"later": ["earlier", "middle"], "middle": ["earlier"]},
        )
        self.assertTrue(queue.require_replan(
            earlier.request_id, "placement_changed"
        ))
        queue.retire_replan(earlier.request_id)
        queue.admit(earlier, 3)

        self.assertEqual(
            queue.snapshot()["causal_predecessors"],
            {"later": ["earlier", "middle"], "middle": ["earlier"]},
        )
        epoch_ns = time.monotonic_ns() - 10_000_000
        for request_id in ("earlier", "middle", "later"):
            self.assertEqual(
                queue.wait(request_id, epoch_ns).status, "ACQUIRED"
            )
            queue.complete(request_id, 101_000)

    def test_replan_preserves_projected_predecessor_order(self) -> None:
        queue = RuntimeDispatchQueue()
        predecessor = decision("predecessor", 10_000, 20_000)
        successor = decision("successor", 20_000, 30_000)
        queue.admit(predecessor, 0)
        queue.admit(successor, 1)

        self.assertTrue(queue.require_replan(
            predecessor.request_id, "placement_changed"
        ))
        self.assertEqual(
            queue.dependent_queued_requests(predecessor.request_id),
            (successor.request_id,),
        )
        self.assertTrue(queue.require_replan(
            successor.request_id, "predecessor_replan"
        ))
        queue.retire_replan(predecessor.request_id)
        queue.admit(decision("predecessor", 400_000, 500_000), 2)
        queue.retire_replan(successor.request_id)
        queue.admit(decision("successor", 500_000, 600_000), 2)

        epoch_ns = time.monotonic_ns() - 400_000_000
        successor_receipts = []
        waiter = threading.Thread(
            target=lambda: successor_receipts.append(
                queue.wait(successor.request_id, epoch_ns)
            )
        )
        waiter.start()
        time.sleep(0.01)
        self.assertTrue(waiter.is_alive())

        acquired = queue.wait(predecessor.request_id, epoch_ns)
        self.assertEqual(acquired.status, "ACQUIRED")
        queue.complete(predecessor.request_id, 101_000)
        waiter.join(timeout=1)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(len(successor_receipts), 1)
        self.assertEqual(successor_receipts[0].status, "ACQUIRED")
        queue.complete(successor.request_id, 102_000)

    def test_replan_removes_obsolete_disjoint_lane_dependency(self) -> None:
        queue = RuntimeDispatchQueue()
        first = decision(
            "first", 0, 20_000,
            route_id="desktop-gpu",
            resource_id="cuda0",
            lane=0,
        )
        second = decision(
            "second", 0, 20_000,
            route_id="desktop-gpu",
            resource_id="cuda0",
            lane=0,
        )
        queue.admit(first, 0)
        queue.admit(second, 0)
        self.assertTrue(queue.require_replan(
            first.request_id, "residency_changed"
        ))
        self.assertTrue(queue.require_replan(
            second.request_id, "predecessor_replan"
        ))
        queue.retire_replan(first.request_id)
        queue.admit(first, 1)
        queue.retire_replan(second.request_id)
        queue.admit(replace(
            second,
            leases=(replace(second.leases[0], lanes=(1,)),),
        ), 1)

        epoch_ns = time.monotonic_ns() - 1_000_000
        self.assertEqual(queue.wait("first", epoch_ns).status, "ACQUIRED")
        receipts = []
        waiter = threading.Thread(
            target=lambda: receipts.append(queue.wait("second", epoch_ns))
        )
        waiter.start()
        waiter.join(0.1)
        acquired_concurrently = not waiter.is_alive()
        queue.complete("first", 2_000)
        waiter.join(1)
        self.assertFalse(waiter.is_alive())
        self.assertTrue(acquired_concurrently)
        self.assertEqual(receipts[0].status, "ACQUIRED")
        queue.complete("second", 2_000)

    def test_residency_follower_replans_after_predecessor_is_queued(
        self,
    ) -> None:
        queue = RuntimeDispatchQueue()
        owner = replace(
            decision("owner", 0, 20_000, resource_id="cuda0"),
            leases=(replace(
                decision("owner", 0, 20_000, resource_id="cuda0")
                .leases[0],
                lanes=(0, 1),
            ),),
        )
        first = replace(
            decision("first", 20_000, 40_000, resource_id="cuda0"),
            leases=(replace(
                decision("first", 20_000, 40_000, resource_id="cuda0")
                .leases[0],
                lanes=(0, 1),
            ),),
        )
        second = replace(
            decision("second", 40_000, 60_000, resource_id="cuda0"),
            leases=(replace(
                decision("second", 40_000, 60_000, resource_id="cuda0")
                .leases[0],
                lanes=(0, 1),
            ),),
        )
        queue.admit(owner, 0)
        queue.admit(first, 1)
        queue.admit(second, 2)
        epoch_ns = time.monotonic_ns() - 1_000_000
        self.assertEqual(queue.wait("owner", epoch_ns).status, "ACQUIRED")
        self.assertTrue(queue.require_replan(
            "first",
            "residency_transition_completed",
            defer_behind_predecessors=True,
        ))
        self.assertTrue(queue.require_replan(
            "second",
            "residency_transition_completed",
            defer_behind_predecessors=True,
        ))
        queue.complete("owner", 1_000)
        first_wake = queue.wait_ready("first", epoch_ns)
        self.assertEqual(first_wake.status, "REPLAN_REQUIRED")
        queue.retire_replan(
            "first", first_wake.queue_generation
        )
        queue.admit(replace(
            first,
            leases=(replace(first.leases[0], lanes=(0,)),),
        ), 1_001)

        receipts = []
        waiter = threading.Thread(
            target=lambda: receipts.append(
                queue.wait_ready("second", epoch_ns)
            ),
            daemon=True,
        )
        waiter.start()
        waiter.join(0.1)

        self.assertTrue(waiter.is_alive())
        self.assertEqual(receipts, [])
        self.assertEqual(queue.wait("first", epoch_ns).status, "ACQUIRED")
        queue.complete("first", 1_002)
        waiter.join(0.1)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(receipts[0].status, "REPLAN_REQUIRED")

    def test_replan_dependencies_follow_scheduled_order(self) -> None:
        queue = RuntimeDispatchQueue()
        successor = decision("successor", 30_000, 40_000)
        predecessor = decision("predecessor", 10_000, 20_000)
        queue.admit(successor, 0)
        queue.admit(predecessor, 1)
        self.assertTrue(queue.require_replan(
            predecessor.request_id, "placement_changed"
        ))

        self.assertEqual(
            queue.dependent_queued_requests(predecessor.request_id),
            (successor.request_id,),
        )

    def test_replan_projection_follows_scheduled_order(self) -> None:
        queue = RuntimeDispatchQueue()
        successor = decision("successor", 30_000, 40_000)
        predecessor = decision("predecessor", 10_000, 20_000)
        queue.admit(successor, 0)
        queue.admit(predecessor, 1)
        self.assertTrue(queue.require_replan(
            successor.request_id, "placement_changed"
        ))

        self.assertEqual(
            queue.preceding_scheduled_requests(successor.request_id),
            (predecessor.request_id,),
        )

    def test_checkpoint_supports_immutable_decision_mappings(self) -> None:
        queue = RuntimeDispatchQueue()
        immutable = replace(
            decision("r0", 0, 20_000),
            queue_by_resource_us=MappingProxyType({"op15-adreno": 0}),
            energy_breakdown=MappingProxyType({"phone_system": 10}),
        )
        queue.admit(immutable, 0)

        checkpoint = queue.checkpoint()
        self.assertEqual(
            queue.wait("r0", time.monotonic_ns() - 1_000_000).status,
            "ACQUIRED",
        )
        queue.restore(checkpoint)

        snapshot = queue.snapshot()
        self.assertEqual(snapshot["active"], {})
        self.assertEqual(snapshot["queued"], {"phone-adreno": ["r0"]})
        self.assertEqual(
            queue.wait("r0", time.monotonic_ns() - 1_000_000).status,
            "ACQUIRED",
        )
        queue.complete("r0", 2_000)

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

    def test_busy_phone_expired_lease_replans_on_completion(self) -> None:
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
        self.assertEqual(receipt[0].status, "REPLAN_REQUIRED")
        self.assertEqual(
            receipt[0].wake_reason,
            "lease_coverage_expired_before_dispatch",
        )
        queue.retire_replan("r1")

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

    def test_lease_extension_coalesces_replan_until_completion(self) -> None:
        queue = RuntimeDispatchQueue()
        active = decision("r0", 0, 10_000)
        queued = decision("r1", 10_000, 20_000)
        epoch_ns = time.monotonic_ns() - 1_000_000
        queue.admit(active, 0)
        queue.admit(queued, 0)
        self.assertEqual(queue.wait("r0", epoch_ns).status, "ACQUIRED")
        receipts = []
        waiter = threading.Thread(
            target=lambda: receipts.append(queue.wait("r1", epoch_ns))
        )
        waiter.start()
        time.sleep(0.01)

        request_ids, committed = queue.commit_conflicting_replans(
            "r0",
            active.leases,
            15_000,
            "lease_upper_bound_overrun",
            lambda values: values,
        )
        self.assertEqual(request_ids, ("r1",))
        self.assertEqual(committed, request_ids)
        time.sleep(0.01)
        self.assertTrue(waiter.is_alive())
        self.assertEqual(receipts, [])

        queue.complete("r0", 15_000)
        waiter.join(1)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(receipts[0].status, "REPLAN_REQUIRED")
        self.assertEqual(
            receipts[0].wake_reason, "predecessor_completion"
        )
        queue.retire_replan("r1")

    def test_lease_extension_cancels_replanning_follower(self) -> None:
        queue = RuntimeDispatchQueue()
        active = decision("r0", 0, 10_000)
        queued = decision("r1", 10_000, 20_000)
        epoch_ns = time.monotonic_ns() - 1_000_000
        queue.admit(active, 0)
        queue.admit(queued, 0)
        self.assertEqual(queue.wait("r0", epoch_ns).status, "ACQUIRED")
        self.assertTrue(queue.require_replan("r1", "projection_changed"))
        receipt = queue.wait("r1", epoch_ns)
        self.assertEqual(receipt.status, "REPLAN_REQUIRED")
        queue.retire_replan("r1", receipt.queue_generation)

        request_ids, committed = queue.commit_conflicting_replans(
            "r0",
            active.leases,
            15_000,
            "lease_upper_bound_overrun",
            lambda values: values,
        )

        self.assertEqual(request_ids, ("r1",))
        self.assertEqual(committed, request_ids)
        snapshot = queue.snapshot()
        self.assertEqual(
            snapshot["entry_states"]["r1"]["state"],
            "DEFERRED_REPLAN",
        )
        self.assertEqual(
            snapshot["entry_states"]["r1"]["wake_reason"],
            "lease_upper_bound_overrun",
        )

    def test_dispatch_wake_waits_for_scheduler_publication(self) -> None:
        queue = RuntimeDispatchQueue()
        queue.admit(decision("r0", 0, 10_000), 0)
        receipts = []
        with queue.defer_wake():
            self.assertTrue(
                queue.require_replan("r0", "learning_generation_changed")
            )
            waiter = threading.Thread(
                target=lambda: receipts.append(
                    queue.wait("r0", time.monotonic_ns())
                )
            )
            waiter.start()
            time.sleep(0.01)
            self.assertTrue(waiter.is_alive())
            self.assertEqual(receipts, [])
        waiter.join(1)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(receipts[0].status, "REPLAN_REQUIRED")
        self.assertEqual(
            receipts[0].wake_reason, "learning_generation_changed"
        )
        queue.retire_replan("r0")

    def test_later_conflict_waits_for_earlier_replan_commit(self) -> None:
        queue = RuntimeDispatchQueue()
        epoch_ns = time.monotonic_ns() - 1_000_000
        queue.admit(decision("active", 0, 100_000), 0)
        queue.admit(decision("same-model", 0, 100_000), 0)
        queue.admit(decision("replacement", 0, 100_000), 0)
        self.assertEqual(
            queue.wait("active", epoch_ns).status, "ACQUIRED"
        )
        self.assertTrue(
            queue.require_replan("same-model", "predecessor_completion")
        )
        replan = queue.wait("same-model", epoch_ns)
        self.assertEqual(replan.status, "REPLAN_REQUIRED")

        receipts = []
        waiter = threading.Thread(
            target=lambda: receipts.append(
                queue.wait("replacement", epoch_ns)
            )
        )
        waiter.start()
        queue.complete("active", 2_000)
        time.sleep(0.01)

        self.assertTrue(waiter.is_alive())
        self.assertEqual(receipts, [])

        queue.retire_replan("same-model")
        queue.admit(decision("same-model", 0, 100_000), 2_000)
        self.assertEqual(
            queue.wait("same-model", epoch_ns).status, "ACQUIRED"
        )
        queue.complete("same-model", 3_000)
        waiter.join(1)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(receipts[0].status, "ACQUIRED")
        queue.complete("replacement", 4_000)

    def test_completion_replan_waits_for_last_blocker(self) -> None:
        queue = RuntimeDispatchQueue()
        cpu = decision(
            "cpu-owner", 0, 10_000,
            route_id="desktop-cpu",
            resource_id="desktop-cpu",
        )
        phone = decision(
            "phone-owner", 0, 10_000,
            route_id="phone-full",
            resource_id="phone-link",
        )
        follower = replace(
            decision(
                "composite-follower", 10_000, 20_000,
                route_id="cpu-phone",
                resource_id="desktop-cpu",
            ),
            leases=(
                LeaseRecord(
                    token="lease-composite-cpu",
                    owner_id="composite-follower",
                    lease_id="cpu-phone-task",
                    resource_id="desktop-cpu",
                    lanes=(0,),
                    start_us=10_000,
                    predicted_end_us=19_990,
                    reserved_until_us=20_000,
                ),
                LeaseRecord(
                    token="lease-composite-phone",
                    owner_id="composite-follower",
                    lease_id="cpu-phone-task",
                    resource_id="phone-link",
                    lanes=(0,),
                    start_us=10_000,
                    predicted_end_us=19_990,
                    reserved_until_us=20_000,
                ),
            ),
        )
        queue.admit(cpu, 0)
        queue.admit(phone, 0)
        queue.admit(follower, 0)
        epoch_ns = time.monotonic_ns() - 1_000_000
        self.assertEqual(queue.wait("cpu-owner", epoch_ns).status, "ACQUIRED")
        self.assertEqual(queue.wait("phone-owner", epoch_ns).status, "ACQUIRED")

        self.assertEqual(
            queue.conflicting_queued_requests(cpu, 2**63 - 1),
            (),
        )
        queue.complete("cpu-owner", 2_000)
        self.assertEqual(
            queue.conflicting_queued_requests(phone, 2**63 - 1),
            ("composite-follower",),
        )

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
    def test_slow_endpoint_does_not_starve_phone_refresh(self) -> None:
        release = threading.Event()
        entered = threading.Event()
        calls = []

        def slow():
            entered.set()
            release.wait(2)

        def phone():
            calls.append(time.monotonic_ns())
            return {"available_bytes": 123}

        monitor = BackgroundRuntimeMonitor(
            {"endpoint": slow, "phone": phone},
            refresh_interval_s=0.01, stale_after_s=0.1,
        )
        monitor.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertTrue(monitor.wait_until_populated(("phone",), 0.5))
            initial = monitor.snapshot("phone").captured_at_ns
            monitor.request_refresh("phone")
            deadline = time.monotonic() + 0.5
            while monitor.snapshot("phone").captured_at_ns == initial:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.001)
            self.assertFalse(monitor.snapshot("phone").stale)
            self.assertFalse(release.is_set())
        finally:
            release.set()
            monitor.stop()

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
