#!/usr/bin/env python3

from __future__ import annotations

import copy
import subprocess
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent / "s15_runtime_dispatch", HERE.parent / "s15_live_launcher"):
    sys.path.insert(0, str(path))

import arrival_launcher as launcher  # noqa: E402
import validate_arrival_result as result_validator  # noqa: E402


BOOT = "test-boot"


def request() -> dict:
    return {
        "schema": "s15-physical-request-v1",
        "command": "EXECUTE",
        "protocol_version": 1,
        "launch_id": 1,
        "route_id": "op15-gemma-head-0-8",
        "profile_id": launcher.EXPECTED_PROFILE,
        "device_id": "op15:3C15AU002CL00000",
        "route_epoch": 13,
        "residency_epoch": 1,
        "lease_epoch": 1,
        "device_boot_epoch": 1,
        "registry_generation": 1,
        "compatibility_key": "gemma-4-12b-it-f16|decode|gemma-head-0-8",
        "request_ids": [f"r{index:02d}" for index in range(32)],
        "cohort_sha256": "sha256:85e0614d31a6f35ad7d09c3b35b2ab4fa9ef2916bc4a1dcfc32d6f4c70a51ec4",
        "input_manifest_sha256": "sha256:ea1f2d47b8c6dec6096c8b95b6073181ecba6838d97741ca92ec2ceca6937858",
        "timeout_us": 4_000_000,
        "expected_boundary_schema": "s15-boundary-certificate-v1",
        "worker_binary_sha256": "sha256:2494f191515ca576cacd8c411de79fa8985f719cba2a11cd6371aa6e1024be9f",
        "worker_generation": 1,
        "device_boot_id": BOOT,
        "layer_range": [0, 8],
    }


class ArrivalLauncherTests(unittest.TestCase):
    def test_exact_request_is_accepted(self) -> None:
        launcher.validate_request(request(), BOOT)

    def test_generic_five_second_timeout_is_rejected(self) -> None:
        value = copy.deepcopy(request())
        value["timeout_us"] = 5_000_000
        with self.assertRaisesRegex(launcher.ArrivalLauncherError, "timeout_us"):
            launcher.validate_request(value, BOOT)

    def test_post_load_profile_is_exact(self) -> None:
        self.assertEqual(
            launcher.EXPECTED_PROFILE,
            "sha256:6fea9922ca8a47779d07d235d0512b25661b176c643e82eb612811519b88d297",
        )
        point = next(
            item for item in launcher.EXPECTED_SNAPSHOT.config.points if item.batch_size == 32
        )
        self.assertEqual(point.duration_us, 3_968_367)

    def test_top_level_runner_refuses_without_run(self) -> None:
        process = subprocess.run(
            [sys.executable, str(HERE / "run_arrival_once.py")],
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("physical execution requires --run", process.stderr)

    def test_persisted_physical_result_validates(self) -> None:
        report = result_validator.validate()
        self.assertEqual(report["verdict"], "ARRIVAL_FAITHFUL_PHYSICAL_B32_PASS")
        self.assertEqual(report["deadline_margin_us"], 422_865)

    def test_transport_cannot_borrow_the_prelaunch_wait(self) -> None:
        report = {"transport_elapsed_us": 4_000_001}
        cohort = {
            "admission_schedule": {
                "planned_launch_us": 248_000_000,
                "earliest_deadline_us": 252_000_000,
            }
        }
        paid = {"elapsed_us": 3_500_000}
        transport = {"state": "reply", "elapsed_us": 4_000_001}
        with self.assertRaisesRegex(result_validator.ValidationError, "full response timing"):
            result_validator.validate_transport_timing(report, cohort, paid, transport)

    def test_deadline_margin_is_derived(self) -> None:
        report = {
            "transport_elapsed_us": 3_500_000,
            "logical_launch_us": 248_000_000,
            "logical_finish_us": 251_500_000,
            "earliest_deadline_us": 252_000_000,
            "deadline_margin_us": 500_001,
        }
        cohort = {
            "admission_schedule": {
                "planned_launch_us": 248_000_000,
                "earliest_deadline_us": 252_000_000,
            }
        }
        paid = {"elapsed_us": 3_400_000}
        transport = {"state": "reply", "elapsed_us": 3_500_000}
        with self.assertRaisesRegex(result_validator.ValidationError, "logical replay deadline"):
            result_validator.validate_transport_timing(report, cohort, paid, transport)


if __name__ == "__main__":
    unittest.main()
