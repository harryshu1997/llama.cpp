#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import unittest

from analyze_shape_balance_abba import analyze
from policy_adapter import (
    DEFAULT_CALIBRATION,
    materialize_gemma_policy,
)


HERE = Path(__file__).resolve().parent


class ShapeBalanceAdapterTests(unittest.TestCase):
    def test_qualified_policy_is_unchanged(self) -> None:
        policy, balance, calibration_hash = materialize_gemma_policy(
            "qualified"
        )
        self.assertEqual(policy.table, "1:6144,16:6144,512:0")
        self.assertIsNone(balance)
        self.assertIsNone(calibration_hash)

    def test_shadow_policy_is_fitted_by_unified_scheduler(self) -> None:
        policy, balance, calibration_hash = materialize_gemma_policy(
            "shape-balanced", DEFAULT_CALIBRATION
        )
        self.assertEqual(
            policy.table,
            "4:6144,6:5632,8:5120,10:6144,11:5120,"
            "13:6144,14:5120,15:6144,16:5120,512:0",
        )
        self.assertIsNotNone(balance)
        self.assertGreater(balance.predicted_saving_ppm, 0)
        self.assertEqual(len(calibration_hash), 64)

    def test_archived_physical_abba_reanalyzes_exactly(self) -> None:
        root = HERE / "results/physical_abba_v1"
        expected = json.loads(
            (root / "SHAPE_BALANCE_ABBA.json").read_text(encoding="ascii")
        )
        actual = analyze({
            "fixed_r1": root / "fixed-r1",
            "tuned_r1": root / "tuned-r1",
            "tuned_r2": root / "tuned-r2",
            "fixed_r2": root / "fixed-r2",
        })
        self.assertEqual(actual, expected)
        self.assertEqual(actual["admission"], "RETAIN_FIXED_POLICY")


if __name__ == "__main__":
    unittest.main()
