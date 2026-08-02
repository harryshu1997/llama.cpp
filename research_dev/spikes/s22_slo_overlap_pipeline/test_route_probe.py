#!/usr/bin/env python3

from __future__ import annotations

import unittest

from route_probe import nearest_rank


class RouteProbeTests(unittest.TestCase):
    def test_nearest_rank(self) -> None:
        values = [4.0, 1.0, 3.0, 2.0]
        self.assertEqual(nearest_rank(values, 1, 2), 2.0)
        self.assertEqual(nearest_rank(values, 19, 20), 4.0)

    def test_nearest_rank_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            nearest_rank([], 1, 2)


if __name__ == "__main__":
    unittest.main()
