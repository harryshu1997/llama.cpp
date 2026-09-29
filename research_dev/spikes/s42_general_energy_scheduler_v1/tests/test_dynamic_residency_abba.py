#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
for path in (REPO_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_residency_v1.analyze_dual_residency_abba import (  # noqa: E402
    DEFAULT_ROOTS,
    analyze,
)
from dynamic_residency_v1.analyze_dual_residency_pair import (  # noqa: E402
    canonical,
    read_object,
)


RESULT = (
    ROOT
    / "dynamic_residency_v1/results/DUAL_RESIDENCY_SERVICE_ABBA_V1.json"
)


class DynamicResidencyAbbaTests(unittest.TestCase):
    def test_checked_in_result_matches_analyzer(self) -> None:
        self.assertEqual(read_object(RESULT), analyze(DEFAULT_ROOTS))

    def test_both_pairs_have_lower_energy_and_wall_time(self) -> None:
        result = analyze(DEFAULT_ROOTS)
        self.assertEqual(
            result["status"],
            "REPEATED_SERVICE_DIRECTION_PASS_NO_ADMISSION",
        )
        self.assertEqual(
            result["run_order"],
            ["control_r1", "treatment_r1", "treatment_r2", "control_r2"],
        )
        self.assertEqual(
            result["pair_server_energy_saving_pct"],
            [9.651141891305992, 7.654245742097842],
        )
        self.assertEqual(
            result["pair_wall_service_saving_pct"],
            [5.511601553245349, 8.911252160557781],
        )
        self.assertAlmostEqual(
            result["changes"]["server_energy_pct"], -8.655677702193309
        )
        self.assertAlmostEqual(
            result["changes"]["wall_service_pct"], -7.247610533708205
        )

    def test_cpu_saving_exceeds_added_gpu_energy(self) -> None:
        result = analyze(DEFAULT_ROOTS)
        changes = result["changes"]
        self.assertAlmostEqual(
            changes["cpu_package_energy_pct"], -24.055298494840393
        )
        self.assertAlmostEqual(
            changes["gpu_board_energy_pct"], 45.0473367282296
        )
        self.assertLess(changes["server_energy_pct"], 0)
        self.assertEqual(
            result["control_mean_of_two"]["server_energy"],
            7053.3174005,
        )
        self.assertEqual(
            result["treatment_mean_of_two"]["server_energy"],
            6442.8049790000005,
        )

    def test_latency_and_quality_regressions_are_visible(self) -> None:
        result = analyze(DEFAULT_ROOTS)
        self.assertFalse(
            result["screen_gates"]["qwen_first_token_not_regressed"]
        )
        self.assertAlmostEqual(
            result["changes"]["qwen_first_token_pct"], 8.903877356554712
        )
        self.assertEqual(result["quality"]["qwen"]["exact"], True)
        self.assertEqual(result["quality"]["gemma"]["exact"], False)
        self.assertEqual(
            result["quality"]["gemma"]["positional_matches"], 37
        )

    def test_repeated_screen_does_not_admit_dynamic_policy(self) -> None:
        result = analyze(DEFAULT_ROOTS)
        self.assertFalse(result["admission"]["eligible"])
        self.assertIsNone(result["admission"]["dynamic_energy_claim"])
        self.assertTrue(result["claim_gates"]["service_abba_repeated"])
        self.assertFalse(result["claim_gates"]["full_trace_equal_work"])
        self.assertFalse(
            result["claim_gates"]["load_transition_restore_energy_included"]
        )
        self.assertFalse(result["claim_gates"]["op15_energy_included"])
        self.assertFalse(
            result["claim_gates"]["protected_gpu_fence_receipts"]
        )

    def test_record_hash_is_canonical(self) -> None:
        result = analyze(DEFAULT_ROOTS)
        claimed = result.pop("record_sha256")
        self.assertEqual(claimed, hashlib.sha256(canonical(result)).hexdigest())


if __name__ == "__main__":
    unittest.main()
