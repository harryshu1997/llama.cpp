#!/usr/bin/env python3

from __future__ import annotations

import unittest

from dynamic_cut_gate import MIN_CUT_SEPARATION_REL_L2, _digest


class DynamicCutHelpersTests(unittest.TestCase):
    def test_digest_is_stable(self) -> None:
        self.assertEqual(_digest((1.0, 2.0)), _digest((1.0, 2.0)))
        self.assertNotEqual(_digest((1.0, 2.0)), _digest((1.0, 3.0)))

    def test_separation_gate_is_positive(self) -> None:
        self.assertGreater(MIN_CUT_SEPARATION_REL_L2, 0.0)


if __name__ == "__main__":
    unittest.main()
