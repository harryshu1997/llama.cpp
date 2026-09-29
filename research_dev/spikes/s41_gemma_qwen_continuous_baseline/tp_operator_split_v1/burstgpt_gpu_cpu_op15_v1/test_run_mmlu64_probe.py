#!/usr/bin/env python3

import unittest

import run_mmlu64_probe


class MMLUProbeTest(unittest.TestCase):
    def test_prompt_matches_frozen_format(self):
        item = {"question": "Q?", "choices": ["a", "b", "c", "d"]}
        self.assertEqual(
            run_mmlu64_probe.prompt_for(item),
            "Question: Q?\nA. a\nB. b\nC. c\nD. d\n"
            "Answer with exactly one uppercase letter: A, B, C, or D.\n"
            "Answer:",
        )

    def test_answer_parser(self):
        self.assertEqual(run_mmlu64_probe.parse_answer(" C\n"), "C")
        self.assertIsNone(run_mmlu64_probe.parse_answer("answer C"))
        self.assertIsNone(run_mmlu64_probe.parse_answer("AB"))

    def test_terminal_summary_ignores_ready_line(self):
        value = run_mmlu64_probe.terminal_ffn_summary([
            "S41SERVERFFN ready host=127.0.0.1",
            'S41SERVERFFN {"calls":1,"status":"ok"}',
        ], True)
        self.assertEqual(value, {"calls": 1, "status": "ok"})


if __name__ == "__main__":
    unittest.main()
