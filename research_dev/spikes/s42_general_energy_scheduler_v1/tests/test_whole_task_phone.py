#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "whole_task_phone_v1/analyze_resident_route_campaign.py"
SPEC = importlib.util.spec_from_file_location("whole_task_phone_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


class WholeTaskPhoneTests(unittest.TestCase):
    def result(self, mismatch: bool = False) -> dict[str, object]:
        shapes = ((2, 100, 20), (50, 200, 40), (99, 300, 80))
        routes = {
            "desktop-cpu": {
                "duration": lambda inp, out: 1.0 + inp / 1000 + out / 100,
                "energy": lambda inp, out: 30.0 + inp / 100 + out / 50,
            },
            "desktop-cuda": {
                "duration": lambda inp, out: 0.2 + inp / 5000 + out / 1000,
                "energy": lambda inp, out: 18.0 + inp / 500 + out / 200,
            },
            "phone-adreno": {
                "duration": lambda inp, out: 0.5 + inp / 1500 + out / 200,
                "energy": lambda inp, out: 10.0 + inp / 1000 + out / 500,
            },
        }
        cases: list[dict[str, object]] = []
        for repetition in range(1, 4):
            cases.append({
                "accounted_fleet_energy_j": 47.0,
                "cpu_package_average_power_w": 1.0,
                "duration_s": 5.0,
                "gpu_board_average_power_w": 8.0,
                "phone": {"whole_phone_average_power_w": 0.4},
                "route": "idle",
            })
            for route, functions in routes.items():
                for request_index, input_tokens, output_tokens in shapes:
                    digest = f"content-{request_index}"
                    if mismatch and route == "phone-adreno":
                        digest += "-phone"
                    cases.append({
                        "accounted_fleet_energy_j": functions["energy"](
                            input_tokens, output_tokens
                        ),
                        "duration_s": functions["duration"](
                            input_tokens, output_tokens
                        ),
                        "input_tokens": input_tokens,
                        "mixed_request_index": request_index,
                        "output_tokens": output_tokens,
                        "response": {"content_sha256": digest},
                        "route": route,
                    })
        for index, row in enumerate(cases):
            row["paid_start_monotonic_ns"] = index * 10_000_000_000
            row["paid_end_monotonic_ns"] = (
                row["paid_start_monotonic_ns"]
                + round(row["duration_s"] * 1_000_000_000)
            )
        return {
            "cases": cases,
            "schema": ANALYSIS.RESULT_SCHEMA,
            "status": "PASS",
        }

    def write(self, value: object) -> tuple[tempfile.TemporaryDirectory, Path]:
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "RESULT.json"
        path.write_text(json.dumps(value), encoding="ascii")
        return directory, path

    def tail_result(self) -> dict[str, object]:
        return {
            "cases": [
                {
                    "tail_duration_s": 20.0,
                    "tail_gpu_dynamic_energy_j": energy,
                }
                for energy in (10.0, 12.0, 14.0)
            ],
            "model_id": ANALYSIS.MODEL_ID,
            "schema": "s42-cuda-resident-tail-result-v1",
            "status": "PASS",
        }

    def test_profile_selects_measured_whole_phone_route(self) -> None:
        directory, path = self.write(self.result())
        self.addCleanup(directory.cleanup)
        analysis, profile = ANALYSIS.analyze(path)
        self.assertEqual(analysis["quality_class"], "exact")
        self.assertEqual(
            analysis["scheduler_decisions"]["gpu_idle"]["route_id"],
            "phone-adreno",
        )
        self.assertEqual(
            analysis["scheduler_decisions"]["gpu_busy"]["route_id"],
            "phone-adreno",
        )
        self.assertEqual(
            analysis["scheduler_decisions"]["gpu_memory_unavailable"][
                "route_id"
            ],
            "phone-adreno",
        )
        phone = next(
            row for row in profile["routes"] if row["route_id"] == "phone-adreno"
        )
        self.assertEqual(phone["granularity"], "task")
        self.assertEqual(
            set(phone["resource_slots"]), {"op15-adreno", "usb-token-rpc"}
        )
        self.assertEqual(phone["server_busy_ppm"], 20_000)
        self.assertEqual(
            phone["energy"]["boundary_id"], ANALYSIS.BOUNDARY_ID
        )
        self.assertTrue(
            analysis["qualification"]["idle_baseline_subtracted"]
        )

    def test_overlapping_measurement_windows_are_rejected(self) -> None:
        value = self.result()
        measured = [
            row for row in value["cases"] if row["route"] != "idle"
        ]
        measured[1]["paid_start_monotonic_ns"] = measured[0][
            "paid_start_monotonic_ns"
        ]
        directory, path = self.write(value)
        self.addCleanup(directory.cleanup)
        with self.assertRaisesRegex(
            ANALYSIS.AnalysisError, "non-overlapping request windows"
        ):
            ANALYSIS.analyze(path)

    def test_cross_backend_output_drift_downgrades_quality(self) -> None:
        directory, path = self.write(self.result(mismatch=True))
        self.addCleanup(directory.cleanup)
        analysis, profile = ANALYSIS.analyze(path)
        self.assertEqual(analysis["quality_class"], "bounded_numeric")
        self.assertTrue(all(
            row["quality_class"] == "bounded_numeric"
            for row in profile["routes"]
        ))

    def test_cuda_tail_is_charged_once_per_isolated_epoch(self) -> None:
        value = self.result()
        for row in value["cases"]:
            if row["route"] == "desktop-cuda":
                row["accounted_fleet_energy_j"] = (
                    1.0
                    + row["input_tokens"] / 10_000
                    + row["output_tokens"] / 10_000
                )
        directory, path = self.write(value)
        self.addCleanup(directory.cleanup)
        tail_path = Path(directory.name) / "CUDA_TAIL_RESULT.json"
        tail_path.write_text(json.dumps(self.tail_result()), encoding="ascii")

        response_analysis, response_profile = ANALYSIS.analyze(path)
        analysis, isolated_profile = ANALYSIS.analyze(path, tail_path)
        self.assertEqual(
            response_analysis["scheduler_decisions"]["gpu_idle"]["route_id"],
            "desktop-cuda",
        )
        self.assertEqual(
            analysis["scheduler_decisions"]["gpu_idle"]["route_id"],
            "phone-adreno",
        )
        response_cuda = next(
            row
            for row in response_profile["routes"]
            if row["route_id"] == "desktop-cuda"
        )
        isolated_cuda = next(
            row
            for row in isolated_profile["routes"]
            if row["route_id"] == "desktop-cuda"
        )
        self.assertEqual(
            isolated_cuda["energy"]["cost_uj"]["fixed"]
            - response_cuda["energy"]["cost_uj"]["fixed"],
            12_000_000,
        )
        self.assertEqual(
            analysis["scheduler_lifecycle_policy"]["default_if_unknown"],
            "cuda_epoch_open",
        )


if __name__ == "__main__":
    unittest.main()
