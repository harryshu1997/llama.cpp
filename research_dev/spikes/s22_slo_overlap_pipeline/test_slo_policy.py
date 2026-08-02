#!/usr/bin/env python3

from __future__ import annotations

import unittest

from slo_policy import (
    NoFeasibleRoute,
    RouteProfile,
    SloRouter,
    WorkRequest,
)


def profile(
    route_id: str,
    head_name: str,
    offloaded_layers: int,
    step_us: int,
    batch: int = 2,
    active: int = 4,
) -> RouteProfile:
    return RouteProfile(
        route_id, head_name, offloaded_layers, 0, step_us, batch, active,
        50, 2, "sha256:" + "0" * 64,
    )


class SloPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = SloRouter([
            profile("R0", "cuda", 0, 100),
            profile("R1", "op15", 8, 200),
            profile("R2", "op12", 8, 400),
        ])

    def test_tight_medium_and_loose_choose_distinct_routes(self) -> None:
        tight = self.router.admit(WorkRequest(1, 0, 500, 4, 0), 0)
        medium = self.router.admit(WorkRequest(2, 0, 900, 4, 1), 0)
        loose = self.router.admit(WorkRequest(3, 0, 2000, 4, 2), 0)
        self.assertEqual([tight.route_id, medium.route_id, loose.route_id], ["R0", "R1", "R2"])

    def test_route_is_pinned_until_exact_completion(self) -> None:
        decision = self.router.admit(WorkRequest(10, 0, 2000, 4, 0), 0)
        self.assertEqual(self.router.pinned(10), decision)
        with self.assertRaisesRegex(ValueError, "already pinned"):
            self.router.admit(WorkRequest(10, 0, 2000, 4, 0), 0)
        with self.assertRaisesRegex(ValueError, "epoch mismatch"):
            self.router.complete(10, decision.route_epoch + 1)
        self.router.complete(10, decision.route_epoch)
        self.assertIsNone(self.router.pinned(10))

    def test_capacity_forces_another_feasible_route(self) -> None:
        router = SloRouter([
            profile("R1", "op15", 8, 100, batch=1, active=1),
            profile("R0", "cuda", 0, 50, active=2),
        ])
        self.assertEqual(router.admit(WorkRequest(1, 0, 1000, 2, 0), 0).route_id, "R1")
        self.assertEqual(router.admit(WorkRequest(2, 0, 1000, 2, 0), 0).route_id, "R0")

    def test_full_batch_has_no_queue_wave_penalty(self) -> None:
        router = SloRouter([profile("R", "phone", 8, 100, batch=2, active=4)])
        first = router.admit(WorkRequest(1, 0, 1000, 2, 0), 0)
        second = router.admit(WorkRequest(2, 0, 1000, 2, 0), 0)
        third = router.admit(WorkRequest(3, 0, 1000, 2, 0), 0)
        self.assertEqual(first.predicted_finish_us, 200)
        self.assertEqual(second.predicted_finish_us, 200)
        self.assertEqual(third.predicted_finish_us, 400)

    def test_batch_wait_is_bounded_by_slack(self) -> None:
        decision = self.router.admit(WorkRequest(1, 0, 420, 4, 0), 0)
        self.assertEqual(decision.route_id, "R0")
        self.assertEqual(decision.batch_wait_us, 2)

    def test_no_feasible_route_fails_closed(self) -> None:
        with self.assertRaisesRegex(NoFeasibleRoute, "no profiled route"):
            self.router.admit(WorkRequest(1, 0, 10, 4, 0), 0)

    def test_bool_is_not_an_integer(self) -> None:
        with self.assertRaises(TypeError):
            WorkRequest(True, 0, 10, 1, 0)
        with self.assertRaises(TypeError):
            RouteProfile("R", "h", 1, 0, True, 1, 1, 0, 2, "sha256:" + "0" * 64)

    def test_profile_digest_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "digest"):
            RouteProfile("R", "h", 1, 0, 1, 1, 1, 0, 2, "unknown")


if __name__ == "__main__":
    unittest.main()
