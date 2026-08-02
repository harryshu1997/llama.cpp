#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_generation_subset import SubsetError, parse_stop_tokens, select_rows  # noqa: E402


class GenerationSubsetTests(unittest.TestCase):
    def test_selection_spans_eligible_prompt_lengths(self) -> None:
        rows = [
            {
                "event_id": f"e{index}",
                "realized_prompt_tokens": 100 + index * 100,
                "requested_output_tokens": 16,
            }
            for index in range(7)
        ]
        selected = select_rows(rows, 4, 650, 8)
        self.assertEqual(
            [row["realized_prompt_tokens"] for row in selected],
            [100, 300, 400, 600],
        )

    def test_context_filter_and_single_middle(self) -> None:
        rows = [
            {"event_id": "a", "realized_prompt_tokens": 100, "requested_output_tokens": 8},
            {"event_id": "b", "realized_prompt_tokens": 200, "requested_output_tokens": 8},
            {"event_id": "c", "realized_prompt_tokens": 300, "requested_output_tokens": 8},
        ]
        self.assertEqual(select_rows(rows, 1, 250, 8)[0]["event_id"], "b")
        with self.assertRaisesRegex(SubsetError, "not enough"):
            select_rows(rows, 3, 250, 8)

    def test_selection_and_stop_token_types_are_strict(self) -> None:
        with self.assertRaises(ValueError):
            select_rows([], True, 100, 1)
        self.assertEqual(parse_stop_tokens("1,50,106"), (1, 50, 106))
        for value in ("", "1,1", "-1", "x"):
            with self.subTest(value=value), self.assertRaises(SubsetError):
                parse_stop_tokens(value)


if __name__ == "__main__":
    unittest.main()
