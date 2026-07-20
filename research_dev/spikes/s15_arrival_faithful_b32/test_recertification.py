#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import validate_recertification as validator  # noqa: E402


class ValidatorUnitTests(unittest.TestCase):
    def test_frozen_physical_result_validates(self) -> None:
        report = validator.validate()
        self.assertEqual(report["verdict"], "POST_LOAD_B32_ROUTE_CERTIFIED")
        self.assertLessEqual(report["completion_conservative_us"], 4_000_000)

    def test_thermal_stream_parser_rejects_duplicates(self) -> None:
        with self.assertRaisesRegex(validator.ValidationError, "sensor failed"):
            validator.thermal_samples(b"THERMAL nsphmx-0=30000 nsphmx-0=30100\n")

    def test_thermal_stream_parser_rejects_unknown_lines(self) -> None:
        with self.assertRaisesRegex(validator.ValidationError, "invalid record"):
            validator.thermal_samples(b"not-thermal\n")

    def test_placement_rejects_undeclared_cpu_compute(self) -> None:
        cert = {
            "status": "SCHEDULED_PLACEMENT_OK",
            "layer_start": 0,
            "layer_end": 8,
            "missing_buffer_compute_nodes": 0,
            "compute_by_op_and_buffer": {
                "MUL_MAT": {"HTP0": 1},
                "RMS_NORM": {"CPU": 1},
            },
        }
        with self.assertRaisesRegex(validator.ValidationError, "fallback"):
            validator.validate_placement(cert)

    def test_acquisition_refuses_without_explicit_run(self) -> None:
        process = subprocess.run(
            [sys.executable, str(HERE / "run_recertification.py")],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("physical acquisition requires --run", process.stderr)


if __name__ == "__main__":
    unittest.main()
