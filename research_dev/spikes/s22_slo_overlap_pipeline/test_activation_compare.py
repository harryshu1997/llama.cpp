#!/usr/bin/env python3

from __future__ import annotations

import math
import unittest

from activation_compare import classify_trials, compare_trials, compare_vectors


class ActivationCompareTests(unittest.TestCase):
    def test_equal_vectors_are_exact(self) -> None:
        result = compare_vectors([1.0, 2.0], [1.0, 2.0])
        self.assertTrue(result["byte_equal"])
        self.assertEqual(result["rel_l2"], 0.0)
        self.assertEqual(result["cosine"], 1.0)

    def test_vector_error_metrics(self) -> None:
        result = compare_vectors([1.0, 0.0], [1.0, 1.0])
        self.assertFalse(result["byte_equal"])
        self.assertAlmostEqual(result["rel_l2"], 1.0)
        self.assertAlmostEqual(result["cosine"], 1.0 / math.sqrt(2.0))
        self.assertEqual(result["max_abs"], 1.0)

    def test_rejects_non_finite(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare_vectors([1.0], [math.inf])

    def test_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same nonzero width"):
            compare_vectors([1.0], [1.0, 2.0])

    def test_compare_trials_labels_positions(self) -> None:
        rows = compare_trials([[1.0], [2.0]], [[1.0], [3.0]])
        self.assertEqual([row["position"] for row in rows], [0, 1])
        self.assertTrue(rows[0]["byte_equal"])
        self.assertFalse(rows[1]["byte_equal"])

    def test_numeric_pass(self) -> None:
        repeat = [{"byte_equal": True}]
        mode = [{"byte_equal": False, "rel_l2": 0.004, "cosine": 0.9995}]
        self.assertEqual(classify_trials(repeat, mode, 0.005, 0.999), "NUMERIC_PASS")

    def test_numeric_fail(self) -> None:
        repeat = [{"byte_equal": True}]
        mode = [{"byte_equal": False, "rel_l2": 0.006, "cosine": 0.9995}]
        self.assertEqual(classify_trials(repeat, mode, 0.005, 0.999), "NUMERIC_FAIL")

    def test_repeat_must_be_stable(self) -> None:
        repeat = [{"byte_equal": False}]
        mode = [{"byte_equal": True, "rel_l2": 0.0, "cosine": 1.0}]
        self.assertEqual(classify_trials(repeat, mode, 0.005, 0.999), "NONDETERMINISTIC")


if __name__ == "__main__":
    unittest.main()
