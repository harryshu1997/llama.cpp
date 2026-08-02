#!/usr/bin/env python3

from __future__ import annotations

import unittest

from fixed_policy import (
    FixedRouteProfile,
    FixedSloRouter,
    NoFeasibleRoute,
    SloWork,
)


ZERO_DIGEST = "sha256:" + "0" * 64


def profile(
    route_id: str,
    rank: int,
    row_us: int,
    resources: tuple[str, ...],
    max_active: int = 8,
) -> FixedRouteProfile:
    return FixedRouteProfile(
        route_id=route_id,
        offload_rank=rank,
        fixed_us=0,
        prefill_token_us=row_us,
        decode_step_us=row_us,
        profiled_batch=4,
        max_active=max_active,
        gather_cap_us=100,
        resources=resources,
        evidence_sha256=ZERO_DIGEST,
    )


def router(op15_capacity: int = 8) -> FixedSloRouter:
    return FixedSloRouter(
        [
            profile(
                "R0", 0, 100,
                ("cuda-prefix", "cuda-mid", "cuda-tail"),
            ),
            profile(
                "R1", 1, 200,
                ("cuda-prefix", "op15-mid", "cuda-tail"),
            ),
            profile(
                "R2", 2, 400,
                ("op12-prefix", "op15-mid", "cuda-tail"),
            ),
        ],
        {
            "cuda-prefix": 8,
            "cuda-mid": 8,
            "op12-prefix": 8,
            "op15-mid": op15_capacity,
            "cuda-tail": 8,
        },
    )


class FixedPolicyTests(unittest.TestCase):
    def test_tight_medium_and_loose_choose_all_fixed_routes(self) -> None:
        policy = router()
        tight = policy.admit(SloWork(1, 0, 450, 1, 4, 0), 0)
        medium = policy.admit(SloWork(2, 0, 850, 1, 4, 1), 0)
        loose = policy.admit(SloWork(3, 0, 1700, 1, 4, 2), 0)
        self.assertEqual(
            [tight.route_id, medium.route_id, loose.route_id],
            ["R0", "R1", "R2"],
        )

    def test_route_is_pinned_until_exact_completion(self) -> None:
        policy = router()
        decision = policy.admit(SloWork(10, 0, 5000, 1, 4, 0), 0)
        self.assertEqual(policy.pinned(10), decision)
        with self.assertRaisesRegex(ValueError, "already pinned"):
            policy.admit(SloWork(10, 0, 5000, 1, 4, 0), 0)
        with self.assertRaisesRegex(ValueError, "epoch mismatch"):
            policy.complete(10, decision.route_epoch + 1)
        policy.complete(10, decision.route_epoch)
        self.assertIsNone(policy.pinned(10))
        self.assertFalse(any(
            policy.active_counts()["routes"].values()
        ))

    def test_shared_op15_capacity_forces_r0(self) -> None:
        policy = router(op15_capacity=1)
        first = policy.admit(SloWork(1, 0, 5000, 1, 4, 0), 0)
        second = policy.admit(SloWork(2, 0, 5000, 1, 4, 0), 0)
        self.assertEqual(first.route_id, "R2")
        self.assertEqual(second.route_id, "R0")

    def test_batch_wait_is_bounded_by_slack(self) -> None:
        policy = router()
        decision = policy.admit(SloWork(1, 0, 410, 1, 4, 0), 0)
        self.assertEqual(decision.route_id, "R0")
        self.assertEqual(decision.batch_wait_us, 0)

    def test_no_feasible_route_fails_closed(self) -> None:
        with self.assertRaisesRegex(NoFeasibleRoute, "no fixed route"):
            router().admit(SloWork(1, 0, 10, 1, 4, 0), 0)

    def test_profile_set_is_exact(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly R0"):
            FixedSloRouter(
                [profile("R0", 0, 1, ("a",))],
                {"a": 1},
            )

    def test_bool_is_not_an_integer(self) -> None:
        with self.assertRaises(TypeError):
            SloWork(True, 0, 1, 1, 1, 0)


if __name__ == "__main__":
    unittest.main()
