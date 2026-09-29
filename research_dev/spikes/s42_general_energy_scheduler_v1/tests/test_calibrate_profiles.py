#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibrate_profiles import fit_nonnegative_affine  # noqa: E402


class CalibrationTests(unittest.TestCase):
    def test_exact_affine_fit(self) -> None:
        observations = [
            (10, 2, 140),
            (20, 2, 170),
            (10, 5, 161),
            (30, 7, 235),
        ]
        coefficients, metrics = fit_nonnegative_affine(observations)
        self.assertEqual(coefficients, (96, 3, 7))
        self.assertEqual(metrics["max_abs_error_us"], 0)
        self.assertEqual(metrics["max_positive_error_us"], 0)

    def test_negative_unconstrained_term_is_clamped_by_active_set(self) -> None:
        observations = [
            (1, 1, 100),
            (2, 1, 100),
            (3, 2, 120),
            (4, 3, 140),
        ]
        coefficients, _ = fit_nonnegative_affine(observations)
        self.assertTrue(all(value >= 0 for value in coefficients))

    def test_upper_additive_covers_positive_residual(self) -> None:
        observations = [
            (1, 1, 10),
            (2, 1, 20),
            (3, 1, 45),
        ]
        coefficients, metrics = fit_nonnegative_affine(observations)
        for inp, out, actual in observations:
            predicted = coefficients[0] + coefficients[1] * inp + coefficients[2] * out
            self.assertLessEqual(actual, predicted + metrics["max_positive_error_us"])


if __name__ == "__main__":
    unittest.main()
