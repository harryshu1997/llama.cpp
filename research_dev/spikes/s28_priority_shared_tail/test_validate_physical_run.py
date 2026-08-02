#!/usr/bin/env python3

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from validate_physical_run import (
    ValidationError,
    load_json,
    validate_pair,
    validate_worker_log,
)


RESULTS = HERE / "results/physical_20260721T182301Z"
PHONE = HERE / "results/a6000_phones_20260721T182048Z"


class PhysicalPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.control = load_json(RESULTS / "control.json")
        self.treatment = load_json(RESULTS / "treatment.json")

    def test_persisted_pair_passes(self) -> None:
        result = validate_pair(self.control, self.treatment)
        self.assertEqual(result["completed"], 60)
        self.assertGreater(result["cuda_compute_relief_percent"], 0)
        self.assertGreater(result["tail_route_transitions"], 0)

    def test_priority_mix_is_rejected(self) -> None:
        treatment = copy.deepcopy(self.treatment)
        treatment["batch_events"]["cuda-tail"][0]["priorities"] = [0, 1]
        with self.assertRaisesRegex(ValidationError, "priority batch"):
            validate_pair(self.control, treatment)

    def test_missing_cuda_relief_is_rejected(self) -> None:
        treatment = copy.deepcopy(self.treatment)
        treatment["summary"]["summed_cuda_island_compute_us"] = (
            self.control["summary"]["summed_cuda_island_compute_us"]
        )
        with self.assertRaisesRegex(ValidationError, "did not decrease"):
            validate_pair(self.control, treatment)

    def test_urgent_route_mutation_is_rejected(self) -> None:
        treatment = copy.deepcopy(self.treatment)
        urgent = next(
            row for row in treatment["runtime"]["requests"]
            if row["priority"] == 0
        )
        urgent["route_id"] = "R2"
        with self.assertRaisesRegex(ValidationError, "priority policy"):
            validate_pair(self.control, treatment)

    def test_p0_latency_regression_is_rejected(self) -> None:
        treatment = copy.deepcopy(self.treatment)
        treatment["summary"]["priority"]["0"]["latency_us"]["p95"] = (
            self.control["summary"]["priority"]["0"]["latency_us"]["p95"]
            * 2
        )
        with self.assertRaisesRegex(ValidationError, "P0 p95"):
            validate_pair(self.control, treatment)

    def test_tail_non_interleaving_is_rejected(self) -> None:
        treatment = copy.deepcopy(self.treatment)
        treatment["batch_events"]["cuda-tail"].sort(
            key=lambda event: event["contributing_routes"]
        )
        with self.assertRaisesRegex(ValidationError, "did not interleave"):
            validate_pair(self.control, treatment)

    def test_undeclared_phone_cpu_fallback_is_rejected(self) -> None:
        text = (PHONE / "OP12.log").read_text(encoding="utf-8")
        text = text.replace(
            '"GET_ROWS":{"CPU":54}',
            '"HOST_FALLBACK":{"CPU":54}',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "OP12.log"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "undeclared"):
                validate_worker_log(path, "OP12", "HTP0", 0, 8, 0, 200)


if __name__ == "__main__":
    unittest.main()
