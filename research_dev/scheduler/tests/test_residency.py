#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    RESIDENCY_PROBLEM_SCHEMA,
    CohortRoute,
    QueueRequest,
    ResidencyResource,
    ResidencySchedulerError,
    ResidencySwitch,
    optimize_parallel_assignment,
    plan_residency_sequence,
    solve_residency_problem,
)


class ResidencySchedulerTests(unittest.TestCase):
    def test_switch_cost_changes_model_order(self) -> None:
        requests = (
            QueueRequest("a", "a", 0, 1000),
            QueueRequest("b", "b", 0, 1000),
        )
        resource = ResidencyResource("gpu", 1000)
        routes = (
            CohortRoute(
                "a-gpu", "a", "gpu", 1, {"a": 100},
                "a-weights", 800, False,
            ),
            CohortRoute(
                "b-gpu", "b", "gpu", 1, {"b": 100},
                "b-weights", 800, False,
            ),
        )
        switches = (
            ResidencySwitch(
                "gpu", "hot", "a-weights", 10, 0, True,
                ("sha256:hot-a",),
            ),
            ResidencySwitch(
                "gpu", "hot", "b-weights", 100, 0, True,
                ("sha256:hot-b",),
            ),
            ResidencySwitch(
                "gpu", "a-weights", "b-weights", 10, 0, True,
                ("sha256:a-b",),
            ),
            ResidencySwitch(
                "gpu", "b-weights", "a-weights", 100, 0, True,
                ("sha256:b-a",),
            ),
        )
        plan = plan_residency_sequence(
            requests, routes, resource, "hot", 0, switches
        )
        self.assertEqual(plan.model_order, ("a", "b"))
        self.assertEqual(plan.makespan_us, 220)

    def test_unmeasured_switch_fails_closed(self) -> None:
        request = QueueRequest("a", "a", 0, 1000)
        resource = ResidencyResource("gpu", 1000)
        route = CohortRoute(
            "a-gpu", "a", "gpu", 1, {"a": 100},
            "a-weights", 800, False,
        )
        switch = ResidencySwitch(
            "gpu", "hot", "a-weights", 10, 0, False,
            ("sha256:estimated",),
        )
        with self.assertRaisesRegex(
            ResidencySchedulerError, "qualified residency transitions"
        ):
            plan_residency_sequence(
                (request,), (route,), resource, "hot", 0, (switch,)
            )
        plan = plan_residency_sequence(
            (request,), (route,), resource, "hot", 0, (switch,),
            require_measured=False,
        )
        self.assertFalse(plan.measured)

    def test_unknown_switch_energy_remains_unknown(self) -> None:
        request = QueueRequest("a", "a", 0, 1000)
        resource = ResidencyResource("gpu", 1000)
        route = CohortRoute(
            "a-gpu", "a", "gpu", 1, {"a": 100},
            "a-weights", 800, False, {"a": 10},
        )
        switch = ResidencySwitch(
            "gpu", "hot", "a-weights", 10, None, True,
            ("sha256:latency-only",),
        )
        plan = plan_residency_sequence(
            (request,), (route,), resource, "hot", 0, (switch,)
        )
        self.assertIsNone(plan.energy_uj)

    def test_serialized_problem_uses_same_planner(self) -> None:
        problem = {
            "initial_ready_us": 0,
            "initial_residency_id": "hot",
            "requests": [{
                "arrival_us": 0,
                "deadline_us": 1000,
                "model_id": "a",
                "request_id": "a",
            }],
            "resource": {
                "capacity_bytes": 1000,
                "resource_id": "gpu",
            },
            "routes": [{
                "energy_uj": None,
                "model_id": "a",
                "preloaded": False,
                "resident_bytes": 800,
                "residency_id": "a-weights",
                "route_id": "a-gpu",
                "service_us": {"a": 100},
                "slots": 1,
            }],
            "schema": RESIDENCY_PROBLEM_SCHEMA,
            "switches": [{
                "energy_uj": None,
                "evidence_ids": ["sha256:hot-a"],
                "latency_us": 10,
                "measured": True,
                "source_residency_id": "hot",
                "target_residency_id": "a-weights",
            }],
        }
        plan = solve_residency_problem(problem)
        self.assertEqual(plan.model_order, ("a",))
        self.assertEqual(plan.makespan_us, 110)

    def test_serialized_problem_requires_schema(self) -> None:
        with self.assertRaisesRegex(
            ResidencySchedulerError, "schema mismatch"
        ):
            solve_residency_problem({})

    def test_preloaded_phone_salvages_deadline_work(self) -> None:
        requests = (
            QueueRequest("short", "gemma", 0, 50),
            QueueRequest("long", "gemma", 0, 50),
        )
        phone = ResidencyResource("phone", 1000, 900)
        gpu = ResidencyResource("gpu", 2000)
        phone_route = CohortRoute(
            "gemma-phone", "gemma", "phone", 1,
            {"short": 20, "long": 200},
            "gemma-ffn", 800, True,
        )
        gpu_route = CohortRoute(
            "gemma-gpu", "gemma", "gpu", 1,
            {"short": 10, "long": 10},
            "gemma-full", 1500, False,
        )
        plan = optimize_parallel_assignment(
            requests,
            phone_route,
            phone,
            0,
            gpu_route,
            gpu,
            100,
        )
        self.assertEqual(plan.source_request_ids, ("short",))
        self.assertEqual(plan.deadline_misses, 1)
        self.assertEqual(plan.source_schedule.jobs[0].start_us, 0)

    def test_htp_mapping_limit_is_separate_from_phone_dram(self) -> None:
        request = QueueRequest("r", "gemma", 0, 100)
        phone = ResidencyResource("phone", 10_000, 900)
        route = CohortRoute(
            "phone", "gemma", "phone", 1, {"r": 10},
            "weights", 1000, True,
        )
        with self.assertRaisesRegex(
            ResidencySchedulerError, "allocation limit"
        ):
            optimize_parallel_assignment(
                (request,), route, phone, 0, route, phone, 0
            )


if __name__ == "__main__":
    unittest.main()
