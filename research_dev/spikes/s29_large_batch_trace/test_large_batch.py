#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from large_batch_adapter import DEFAULT_KNEES, knees_from_args
from large_batch_profiles import ProfileError, canonical_bytes, load_bundle, write_bundle
from large_batch_runtime import LargeBatchAdmissionController
from priority_policy import WaitDecision


def event(batch: int) -> dict:
    return {"status": "OK", "batch_size": batch, "compute_us": 100}


def calibration() -> dict:
    workers = ("cuda-prefix", "cuda-mid", "op12-prefix", "op15-mid", "cuda-tail")
    active = {
        "R0": {"cuda-prefix", "cuda-mid", "cuda-tail"},
        "R2": {"op12-prefix", "op15-mid", "cuda-tail"},
    }
    points = []
    for route in ("R0", "R2"):
        for batch in (1, 4, 24, 32):
            for rep in range(2):
                points.append({
                    "route_id": route,
                    "batch_size": batch,
                    "rep": rep,
                    "wall_us": 1000 * batch + rep,
                    "max_latency_us": 1000 * batch + rep,
                    "cuda_work_us": (300 if route == "R0" else 100) * batch + rep,
                    "output_tokens": [1, 2, 3, 4],
                    "events": {
                        worker: [event(batch) for _ in range(4)] if worker in active[route] else []
                        for worker in workers
                    },
                })
    return {
        "schema": "s29-route-calibration-v1",
        "status": "CALIBRATION_COMPLETE",
        "shape": {"input_tokens": 1, "output_steps": 4, "context": 16},
        "batches": [1, 4, 24, 32],
        "reps": 2,
        "capacities": {
            "cuda-prefix": 32,
            "cuda-mid": 32,
            "op12-prefix": 32,
            "op15-mid": 32,
            "cuda-tail": 32,
        },
        "points": points,
    }


class ProfileTests(unittest.TestCase):
    def test_profile_round_trip_and_route_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "calibration.json"
            profile = root / "profiles.json"
            source.write_bytes(canonical_bytes(calibration()))
            write_bundle(source, profile)
            routes, capacities, reserve, _bundle = load_bundle(profile)
            by_id = {route.route_id: route for route in routes}
            self.assertEqual([point.batch_size for point in by_id["R0"].points], [1, 4, 24, 32])
            self.assertEqual([point.batch_size for point in by_id["R2"].points], [1, 24, 32])
            self.assertEqual(capacities["cuda-tail"], 32)
            self.assertEqual(reserve["cuda-tail"], 0)

    def test_tampered_profile_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "calibration.json"
            profile = root / "profiles.json"
            source.write_bytes(canonical_bytes(calibration()))
            write_bundle(source, profile)
            value = json.loads(profile.read_text(encoding="ascii"))
            value["routes"][1]["points"][-1]["duration_us"] = 1
            profile.write_bytes(canonical_bytes(value))
            with self.assertRaisesRegex(ProfileError, "canonical"):
                load_bundle(profile)

    def test_failed_calibration_event_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "calibration.json"
            profile = root / "profiles.json"
            value = calibration()
            value["points"][0]["events"]["cuda-prefix"][0]["status"] = "ERROR"
            source.write_bytes(canonical_bytes(value))
            with self.assertRaisesRegex(ProfileError, "failed"):
                write_bundle(source, profile)


class ControllerTests(unittest.TestCase):
    def controller(self, enabled: bool = True) -> LargeBatchAdmissionController:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "calibration.json"
            profile = root / "profiles.json"
            source.write_bytes(canonical_bytes(calibration()))
            write_bundle(source, profile)
            routes, capacities, reserve, _bundle = load_bundle(profile)
        return LargeBatchAdmissionController(
            routes,
            capacities,
            reserve,
            offload_enabled=enabled,
            finite_arrival_end_us=2_000_000,
            urgent_quiet_us=1_000_000,
        )

    @staticmethod
    def enqueue(controller: LargeBatchAdmissionController, count: int, priority: int = 1) -> None:
        from priority_policy import PriorityWork

        for request_id in range(1, count + 1):
            controller.enqueue(PriorityWork(
                request_id=request_id,
                arrival_us=0,
                deadline_us=100_000_000,
                priority=priority,
                input_tokens=1,
                output_steps=4,
            ), 2_000_000)

    def test_full_background_cohort_uses_b32_phone(self) -> None:
        controller = self.controller()
        self.enqueue(controller, 32)
        decision = controller.decide(2_000_000)
        self.assertEqual((decision.route_id, decision.batch_size), ("R2", 32))

    def test_b24_finite_drain_uses_phone(self) -> None:
        controller = self.controller()
        self.enqueue(controller, 24)
        decision = controller.decide(2_000_000)
        self.assertEqual((decision.route_id, decision.batch_size), ("R2", 24))

    def test_small_finite_remainder_uses_cuda(self) -> None:
        controller = self.controller()
        self.enqueue(controller, 2)
        decision = controller.decide(2_000_000)
        self.assertEqual((decision.route_id, decision.batch_size), ("R0", 1))

    def test_urgent_uses_cuda(self) -> None:
        controller = self.controller()
        self.enqueue(controller, 1, priority=0)
        decision = controller.decide(2_000_000)
        self.assertEqual((decision.route_id, decision.batch_size), ("R0", 1))

    def test_recent_urgent_arrival_guards_nonpreemptive_phone_batch(self) -> None:
        controller = self.controller()
        self.enqueue(controller, 1, priority=0)
        urgent = controller.decide(0)
        controller.complete_group(urgent.request_ids, urgent.route_epochs)
        self.enqueue(controller, 32)
        guarded = controller.decide(500_000)
        self.assertIsInstance(guarded, WaitDecision)
        self.assertEqual(guarded.next_wake_us, 1_000_000)
        decision = controller.decide(1_000_000)
        self.assertEqual((decision.route_id, decision.batch_size), ("R2", 32))


class AdapterTests(unittest.TestCase):
    def test_default_knees(self) -> None:
        args = SimpleNamespace()
        self.assertEqual(knees_from_args(args), DEFAULT_KNEES)

    def test_nonpositive_knee_is_rejected(self) -> None:
        args = SimpleNamespace(op12_prefix_knee=0)
        with self.assertRaisesRegex(Exception, "positive"):
            knees_from_args(args)


if __name__ == "__main__":
    unittest.main()
