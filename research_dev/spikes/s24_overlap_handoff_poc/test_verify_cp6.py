#!/usr/bin/env python3

import copy
import unittest

from verify_cp6 import (
    WorkloadGateError,
    arrival_histogram,
    validate_completed_work,
)


def source(request_id, arrival_us, route_id):
    return {
        "request_id": request_id,
        "arrival_us": arrival_us,
        "priority": 1,
        "route_hint": route_id,
        "input_tokens": 3,
        "output_steps": 2,
        "synthetic_token": 2,
        "observed_input_tokens": 3,
        "observed_output_tokens": 2,
        "slo_us": 1000000,
    }


def completed(row):
    return {
        "request_id": row["request_id"],
        "prompt_length": row["input_tokens"],
        "output_steps": row["output_steps"],
        "priority": row["priority"],
        "observed_input_tokens": row["observed_input_tokens"],
        "observed_output_tokens": row["observed_output_tokens"],
        "scheduled_arrival_ns": row["arrival_us"] * 1000,
        "output_tokens": [10, 11],
        "route_id": row["route_hint"],
        "slo_met": True,
    }


class VerifyCp6Tests(unittest.TestCase):
    def fixture(self):
        requests = [source(1, 0, "R1"), source(2, 1000000, "R2")]
        trace = {"requests": requests}
        run = {
            "runtime": {
                "requests": [completed(row) for row in requests],
                "rejected": [],
            },
            "summary": {
                "makespan_us": 10,
                "op15_pooled": {"batch_sizes": [2]},
                "cuda_tail": {"batch_sizes": [2]},
                "cuda_island_compute_us": {"cuda-tail": 5},
            },
        }
        return trace, run

    def test_observed_work_requires_exact_lengths_arrivals_and_routes(self):
        trace, run = self.fixture()
        result = validate_completed_work("observed-context-600", trace, run)
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["route_distribution"], {"R1": 1, "R2": 1})

    def test_arrival_change_is_rejected(self):
        trace, run = self.fixture()
        changed = copy.deepcopy(run)
        changed["runtime"]["requests"][0]["scheduled_arrival_ns"] = 1
        with self.assertRaisesRegex(WorkloadGateError, "request work changed"):
            validate_completed_work("observed-context-600", trace, changed)

    def test_arrival_histogram_is_exact(self):
        trace, _run = self.fixture()
        self.assertEqual(
            arrival_histogram(trace),
            {"0": 1, "1000000": 1},
        )


if __name__ == "__main__":
    unittest.main()
