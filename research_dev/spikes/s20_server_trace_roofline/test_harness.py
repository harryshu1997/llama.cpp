#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


BASE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("s20_analyze", BASE / "analyze_replay.py")
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

REPLAY_SPEC = importlib.util.spec_from_file_location("s20_replay", BASE / "run_replay.py")
assert REPLAY_SPEC is not None and REPLAY_SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(REPLAY_SPEC)
REPLAY_SPEC.loader.exec_module(REPLAY)


class HarnessTests(unittest.TestCase):
    def test_classification(self) -> None:
        self.assertEqual(MOD.classify_counts(2, 0), "COMPUTE_DOMINANT")
        self.assertEqual(MOD.classify_counts(0, 2), "MEMORY_DOMINANT")
        self.assertEqual(MOD.classify_counts(1, 1), "MIXED")
        self.assertEqual(MOD.classify_counts(0, 0), "IDLE_OR_TRANSITION")

    def test_request_intervals(self) -> None:
        events = [
            {"kind": "request_start", "event_id": "r0", "t_ns": 10},
            {"kind": "first_token", "event_id": "r0", "t_ns": 20},
            {"kind": "request_end", "event_id": "r0", "t_ns": 30},
        ]
        intervals = MOD.request_intervals(events)
        self.assertEqual(MOD.phase_counts(intervals, 15), (1, 0))
        self.assertEqual(MOD.phase_counts(intervals, 25), (0, 1))

    def test_percentile(self) -> None:
        self.assertEqual(MOD.percentile([4.0, 1.0, 2.0, 3.0], 0.50), 2.0)
        self.assertEqual(MOD.percentile([4.0, 1.0, 2.0, 3.0], 0.95), 4.0)

    def test_slot_phase_parser_accepts_current_and_legacy_shapes(self) -> None:
        current = [{"is_processing": True, "next_token": [{"n_decoded": 0}]}]
        legacy = [{"is_processing": True, "next_token": {"n_decoded": 4}}]
        self.assertEqual(REPLAY.summarize_slots(current)["prefill_slots"], 1)
        self.assertEqual(REPLAY.summarize_slots(legacy)["decode_slots"], 1)
        with self.assertRaises(REPLAY.ReplayError):
            REPLAY.summarize_slots([
                {"is_processing": True, "next_token": [{"n_decoded": 0}, {}]},
            ])

    def test_replay_cli_constructs(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(BASE / "run_replay.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--request-timeout-s", completed.stdout)

    def test_svg_is_ascii_and_contains_lines(self) -> None:
        point = {
            "t_s": 0.1,
            "dram_total_pct": 60.0,
            "tensor_active_pct": 40.0,
            "sm_issue_pct": 30.0,
            "active_slots": 2,
            "prefill_slots": 1,
            "decode_slots": 1,
            "regime": "MIXED",
        }
        svg = MOD.render_svg([point], {})
        svg.encode("ascii")
        self.assertIn("DRAM read + write", svg)
        self.assertIn("Prefill slots", svg)

    def test_html_is_self_contained(self) -> None:
        svg = '<svg xmlns="http://www.w3.org/2000/svg"></svg>\n'
        page = MOD.render_html(svg)
        page.encode("ascii")
        self.assertIn(svg, page)
        self.assertNotIn('src="server_trace.svg"', page)


if __name__ == "__main__":
    unittest.main()
