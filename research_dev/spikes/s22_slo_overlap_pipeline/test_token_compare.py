#!/usr/bin/env python3

from __future__ import annotations

import unittest

from token_compare import classify_tokens


class TokenCompareTests(unittest.TestCase):
    def test_exact(self) -> None:
        self.assertEqual(classify_tokens([1, 2], [1, 2], [1, 2]), "EXACT")

    def test_mode_divergence(self) -> None:
        self.assertEqual(
            classify_tokens([1, 2], [1, 3], [1, 2]),
            "MODE_DIVERGENCE",
        )

    def test_nondeterministic(self) -> None:
        self.assertEqual(
            classify_tokens([1, 2], [1, 2], [1, 3]),
            "NONDETERMINISTIC",
        )

    def test_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            classify_tokens([1], [1, 2], [1])


if __name__ == "__main__":
    unittest.main()
