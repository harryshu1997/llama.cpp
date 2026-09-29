#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_model_energy  # noqa: E402


class ModelEnergyRunnerTest(unittest.TestCase):
    def test_frozen_case_grid(self) -> None:
        rows = run_model_energy.load_cases(ROOT / "CASES_V1.jsonl")
        self.assertEqual(len(rows), 9)
        self.assertEqual(sum(row["kind"] == "idle" for row in rows), 1)
        self.assertEqual(sum(row["holdout"] for row in rows), 2)
        self.assertEqual(
            {row.get("cohort_size") for row in rows if row["kind"] == "inference"},
            {1, 4, 8},
        )

    def test_request_geometry(self) -> None:
        row = run_model_energy.request_row(7, 128, 32)
        self.assertEqual(row["request_index"], 7)
        self.assertEqual(len(row["prompt_tokens"]), 128)
        self.assertEqual(row["output_tokens"], 32)

    def test_strict_integer_rejects_boolean(self) -> None:
        with self.assertRaises(run_model_energy.run_trace.RunError):
            run_model_energy.strict_int(True, "test", 0)

    def test_cuda_pid_parser(self) -> None:
        self.assertEqual(run_model_energy.parse_cuda_pids("12\n34\n"), {12, 34})
        self.assertEqual(run_model_energy.parse_cuda_pids(""), set())
        with self.assertRaises(run_model_energy.run_trace.RunError):
            run_model_energy.parse_cuda_pids("not-a-pid\n")

    def test_resource_summary(self) -> None:
        rows = [
            {
                "gpu": {"power_mw": 10000, "utilization_pct": 10},
                "hot": {"rss_bytes": 100, "swap_bytes": 0},
                "system": {"available_bytes": 1000},
            },
            {
                "gpu": {"power_mw": 20000, "utilization_pct": 20},
                "hot": {"rss_bytes": 200, "swap_bytes": 0},
                "system": {"available_bytes": 900},
            },
        ]
        value = run_model_energy.resource_summary(rows)
        self.assertEqual(value["gpu_power_w"]["mean"], 15.0)
        self.assertEqual(value["model_rss_max_bytes"], 200)
        self.assertEqual(value["system_available_min_bytes"], 900)


if __name__ == "__main__":
    unittest.main()
