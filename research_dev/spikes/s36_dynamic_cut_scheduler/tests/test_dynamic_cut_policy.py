#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_cut_policy import (  # noqa: E402
    DeviceState,
    DynamicCutPolicy,
    PolicyError,
    RequestWork,
    RouteProfile,
)


def profiles() -> list[RouteProfile]:
    return [
        RouteProfile("cuda-c4", "cuda", 4, 100, 10, True, True),
        RouteProfile("op12-c4", "op12", 4, 200, 20, True, True),
        RouteProfile("op12-c8", "op12", 8, 300, 20, True, True),
        RouteProfile("op15-c4", "op15", 4, 180, 20, True, True),
        RouteProfile("op15-c8", "op15", 8, 250, 20, True, True),
    ]


def states() -> dict[str, DeviceState]:
    return {
        "op12": DeviceState(0, 32, 0, True, {4: 0, 8: 0}),
        "op15": DeviceState(0, 32, 0, True, {4: 0, 8: 0}),
    }


class DynamicCutPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = DynamicCutPolicy(profiles())

    def work(self, priority: int = 2, deadline_us: int = 1_000) -> RequestWork:
        return RequestWork(7, priority, 0, deadline_us)

    def test_priority_zero_uses_cuda(self) -> None:
        decision = self.policy.choose(self.work(priority=0), 0, states())
        self.assertEqual(decision.route_id, "cuda-c4")
        self.assertEqual(decision.reason, "PRIORITY_ZERO_CUDA")

    def test_control_uses_cuda(self) -> None:
        decision = self.policy.choose(self.work(), 0, states(), force_control=True)
        self.assertEqual(decision.route_id, "cuda-c4")
        self.assertEqual(decision.reason, "CONTROL_ALL_CUDA")

    def test_deepest_feasible_cut_wins(self) -> None:
        decision = self.policy.choose(self.work(), 0, states())
        self.assertEqual(decision.route_id, "op15-c8")
        self.assertEqual(decision.cut, 8)

    def test_priority_one_selects_fastest_feasible_phone_route(self) -> None:
        decision = self.policy.choose(self.work(priority=1), 0, states())
        self.assertEqual(decision.route_id, "op15-c4")
        self.assertEqual(decision.reason, "FASTEST_FEASIBLE_PHONE_ROUTE")

    def test_same_cut_queue_affinity_breaks_device_tie(self) -> None:
        device_states = states()
        device_states["op12"] = DeviceState(0, 32, 0, True, {4: 0, 8: 9})
        decision = self.policy.choose(self.work(), 0, device_states)
        self.assertEqual(decision.route_id, "op12-c8")

    def test_normalized_active_load_breaks_tie(self) -> None:
        device_states = states()
        device_states["op15"] = DeviceState(20, 32, 0, True, {4: 0, 8: 0})
        decision = self.policy.choose(self.work(), 0, device_states)
        self.assertEqual(decision.route_id, "op12-c8")

    def test_tight_slo_falls_back_without_waiting(self) -> None:
        decision = self.policy.choose(self.work(deadline_us=199), 0, states())
        self.assertEqual(decision.route_id, "cuda-c4")
        self.assertEqual(decision.reason, "NO_FEASIBLE_PHONE_ROUTE")

    def test_queue_prediction_can_make_phone_infeasible(self) -> None:
        device_states = {
            key: DeviceState(0, 32, 1_000, True, {4: 0, 8: 0})
            for key in ("op12", "op15")
        }
        decision = self.policy.choose(self.work(), 0, device_states)
        self.assertEqual(decision.route_id, "cuda-c4")

    def test_unready_and_full_devices_are_excluded(self) -> None:
        device_states = {
            "op12": DeviceState(0, 32, 0, False, {4: 0, 8: 0}),
            "op15": DeviceState(32, 32, 0, True, {4: 0, 8: 0}),
        }
        self.assertEqual(
            self.policy.choose(self.work(), 0, device_states).route_id,
            "cuda-c4",
        )

    def test_eligible_unmeasured_profile_is_rejected(self) -> None:
        bad = profiles()
        bad[1] = RouteProfile("op12-c4", "op12", 4, 200, 20, False, True)
        with self.assertRaises(PolicyError):
            DynamicCutPolicy(bad)

    def test_missing_server_fallback_is_rejected(self) -> None:
        with self.assertRaises(PolicyError):
            DynamicCutPolicy(profiles()[1:])

    def test_boolean_integer_fields_are_rejected(self) -> None:
        with self.assertRaises(PolicyError):
            RequestWork(True, 1, 0, 100).validate()
        with self.assertRaises(PolicyError):
            DeviceState(False, 32, 0, True, {}).validate()


if __name__ == "__main__":
    unittest.main()
