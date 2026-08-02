#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest

from evaluate_controls import ControlError, evaluate_controls


def report(control: str, repetition: int) -> dict:
    if control == "C0":
        routes = {"R0": 2}
        cuda_us = 1000
        op15_sizes = []
        ttft = 100
        latency = 200
    elif control == "C1":
        routes = {"R0": 1, "R1": 1}
        cuda_us = 900
        op15_sizes = [1, 1]
        ttft = 100
        latency = 200
    elif control == "C2":
        routes = {"R0": 1, "R1": 1}
        cuda_us = 900
        op15_sizes = [2]
        ttft = 104
        latency = 209
    else:
        routes = {"R0": 1, "R2": 1}
        cuda_us = 800
        op15_sizes = [1]
        ttft = 100
        latency = 200
    requests = [
        {
            "request_id": 1,
            "prompt_length": 1,
            "output_steps": 4,
            "observed_input_tokens": 10,
            "observed_output_tokens": 5,
            "scheduled_arrival_ns": 0,
            "priority": 0,
            "ttft_us": ttft,
            "latency_us": latency,
            "slo_met": True,
        },
        {
            "request_id": 2,
            "prompt_length": 1,
            "output_steps": 4,
            "observed_input_tokens": 20,
            "observed_output_tokens": 6,
            "scheduled_arrival_ns": 1000,
            "priority": 1,
            "ttft_us": 300,
            "latency_us": 400,
            "slo_met": True,
        },
    ]
    return {
        "trace": {"trace_hash": "sha256:trace"},
        "runtime": {"requests": requests},
        "summary": {
            "completed_requests": 2,
            "rejected_requests": 0,
            "slo_misses": 0,
            "route_distribution": routes,
            "makespan_us": 1000 + repetition,
            "op15_pooled": {"batch_sizes": op15_sizes},
            "summed_cuda_island_compute_us": cuda_us,
            "gpu_board_energy": {"energy_j": 10.0 + repetition},
        },
    }


def wrapper(control: str, repetition: int) -> dict:
    return {
        "path": f"{control}-{repetition}.json",
        "sha256": "sha256:run",
        "validation_path": f"{control}-{repetition}-validation.json",
        "validation_sha256": "sha256:validation",
        "report": report(control, repetition),
    }


class ControlEvaluationTests(unittest.TestCase):
    def rows(self) -> dict:
        return {
            control: [wrapper(control, repetition) for repetition in range(3)]
            for control in ("C0", "C1", "C2", "C3")
        }

    def test_all_frozen_gates_pass(self) -> None:
        result = evaluate_controls(self.rows())
        self.assertEqual(result["status"], "BENEFIT_GATES_PASS")
        self.assertTrue(result["gates"]["shared_op15_batch_or_makespan"]["pass"])
        self.assertTrue(result["gates"]["priority_0_regression_limit"]["pass"])
        self.assertTrue(result["gates"]["c3_cuda_compute_reduction"]["pass"])

    def test_priority_regression_fails_gate(self) -> None:
        rows = self.rows()
        broken = copy.deepcopy(rows)
        for row in broken["C2"]:
            row["report"]["runtime"]["requests"][0]["ttft_us"] = 106
        result = evaluate_controls(broken)
        self.assertEqual(result["status"], "BENEFIT_GATE_FAIL")
        self.assertFalse(result["gates"]["priority_0_regression_limit"]["pass"])

    def test_arrival_change_is_not_equal_work(self) -> None:
        rows = self.rows()
        rows["C2"][0]["report"]["runtime"]["requests"][0][
            "scheduled_arrival_ns"
        ] = 1
        with self.assertRaisesRegex(ControlError, "completed work differs"):
            evaluate_controls(rows)


if __name__ == "__main__":
    unittest.main()
