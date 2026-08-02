#!/usr/bin/env python3

from __future__ import annotations

import unittest

from batch_knee import parse_candidates, select_knee


class BatchKneeTests(unittest.TestCase):
    def test_smallest_batch_within_five_percent_of_peak_is_selected(self) -> None:
        knee, throughputs = select_knee({
            1: [1000, 1000, 1000],
            2: [1200, 1200, 1200],
            4: [2100, 2100, 2100],
            8: [4200, 4200, 4200],
        })
        self.assertEqual(knee, 4)
        self.assertGreater(throughputs[8], throughputs[2])

    def test_exact_tie_selects_smaller_batch(self) -> None:
        knee, _throughputs = select_knee({
            1: [1000],
            2: [2000],
        })
        self.assertEqual(knee, 1)

    def test_invalid_sample_fails(self) -> None:
        with self.assertRaises(ValueError):
            select_knee({1: []})

    def test_candidate_parser_is_strict(self) -> None:
        self.assertEqual(parse_candidates("1,2,4"), (1, 2, 4))
        with self.assertRaises(Exception):
            parse_candidates("2,1")


if __name__ == "__main__":
    unittest.main()
