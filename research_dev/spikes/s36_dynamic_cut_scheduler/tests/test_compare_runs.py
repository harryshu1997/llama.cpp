#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_runs import compare  # noqa: E402


def run(treatment: bool) -> dict:
    requests = []
    for request_id in range(1, 61):
        requests.append({
            "request_id": request_id,
            "output_tokens": [3, 4, 5, 6],
        })
    route_distribution = (
        {"op12-c4": 20, "op15-c8": 30, "cuda-c4": 10}
        if treatment else {"cuda-c4": 60}
    )
    return {
        "trace": {"trace_hash": "sha256:trace"},
        "profiles": {"profile_hash": "sha256:profile"},
        "runtime": {"requests": requests},
        "summary": {
            "route_distribution": route_distribution,
            "device_distribution": (
                {"op12": 20, "op15": 30, "cuda": 10}
                if treatment else {"cuda": 60}
            ),
            "cut_distribution": (
                {"4": 30, "8": 30} if treatment else {"4": 60}
            ),
            "phone_mixed_phase_batches": 1 if treatment else 0,
            "selected_cuda_compute_us": 800 if treatment else 1_000,
            "priority": {
                "0": {
                    "slo_misses": 0,
                    "latency_us": {"p95": 102 if treatment else 100},
                }
            },
        },
        "final_state": {
            "active_counts": {"cuda": 0, "op12": 0, "op15": 0},
            "tail_active": 0,
            "route_pins": {},
            "software_leases": {"cuda": {}, "op12": {}, "op15": {}, "tail": {}},
        },
    }


class CompareRunTests(unittest.TestCase):
    def test_all_frozen_gates_pass(self) -> None:
        report = compare(run(False), run(True))
        self.assertEqual(report["verdict"], "PASS")
        self.assertAlmostEqual(
            report["gates"]["selected_cuda_compute_relief_percent"], 20.0,
        )

    def test_token_mutation_fails(self) -> None:
        treatment = run(True)
        treatment["runtime"]["requests"][0]["output_tokens"][0] = 99
        self.assertIn("TOKEN_MISMATCH", compare(run(False), treatment)["problems"])

    def test_missing_cut_fails(self) -> None:
        treatment = run(True)
        treatment["summary"]["cut_distribution"] = {"8": 50}
        self.assertIn("BOTH_CUTS_NOT_USED", compare(run(False), treatment)["problems"])

    def test_priority_zero_regression_fails(self) -> None:
        treatment = run(True)
        treatment["summary"]["priority"]["0"]["latency_us"]["p95"] = 106
        self.assertIn(
            "PRIORITY_ZERO_P95_REGRESSION",
            compare(run(False), treatment)["problems"],
        )

    def test_no_cuda_relief_fails(self) -> None:
        treatment = run(True)
        treatment["summary"]["selected_cuda_compute_us"] = 1_000
        self.assertIn(
            "NO_SELECTED_CUDA_COMPUTE_RELIEF",
            compare(run(False), treatment)["problems"],
        )


if __name__ == "__main__":
    unittest.main()
