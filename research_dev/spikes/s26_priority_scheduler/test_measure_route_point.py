#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from measure_route_point import (
    PhysicalRuntimeError,
    build_requests,
    validate_measurement_shape,
)


def report(route_id: str = "R0") -> dict:
    events = {
        "cuda-prefix": [],
        "cuda-mid": [],
        "op12-prefix": [],
        "op15-mid": [],
        "cuda-tail": [],
    }
    active = (
        ("cuda-prefix", "cuda-mid", "cuda-tail")
        if route_id == "R0"
        else ("op12-prefix", "op15-mid", "cuda-tail")
    )
    for name in active:
        events[name] = [{"batch_size": 4, "status": "OK"} for _ in range(4)]
    return {
        "schema": "s24-fixed-diamond-physical-v1",
        "status": "RUN_COMPLETE",
        "control": "C0",
        "route_point": {"route_id": route_id},
        "runtime": {
            "completed_count": 4,
            "rejected_count": 0,
            "requests": [{
                "route_id": route_id,
                "prompt_length": 1,
                "output_steps": 4,
                "slo_met": True,
            } for _ in range(4)],
        },
        "batch_events": events,
    }


class RoutePointTests(unittest.TestCase):
    def test_requests_form_one_homogeneous_b4_group(self) -> None:
        requests = build_requests("R0", 5000)
        self.assertEqual(tuple(row.request_id for row in requests), tuple(range(26401, 26405)))
        self.assertEqual({row.route_id for row in requests}, {"R0"})
        self.assertEqual({row.batch_wait_us for row in requests}, {5000})

    def test_r2_shape_passes(self) -> None:
        validate_measurement_shape(report("R2"))

    def test_r2_uses_low_priority_slo(self) -> None:
        requests = build_requests("R2", 5000)
        self.assertEqual({row.priority for row in requests}, {2})
        self.assertEqual({row.slo_us for row in requests}, {15_000_000})

    def test_unknown_route_is_rejected(self) -> None:
        with self.assertRaisesRegex(PhysicalRuntimeError, "R0 or R2"):
            build_requests("R1", 5000)

    def test_exact_b4_shape_passes(self) -> None:
        validate_measurement_shape(report())

    def test_b1_execution_is_rejected(self) -> None:
        value = report()
        value["batch_events"]["cuda-prefix"][0]["batch_size"] = 1
        with self.assertRaisesRegex(PhysicalRuntimeError, "lockstep B4"):
            validate_measurement_shape(value)

    def test_failed_batch_is_rejected(self) -> None:
        value = report()
        value["batch_events"]["cuda-tail"][0]["status"] = "FAIL"
        with self.assertRaisesRegex(PhysicalRuntimeError, "failed physical"):
            validate_measurement_shape(value)


if __name__ == "__main__":
    unittest.main()
