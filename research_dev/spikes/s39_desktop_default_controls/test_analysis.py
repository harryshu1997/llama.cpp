#!/usr/bin/env python3
"""Tests for independent stock-default reduction."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest


import analyze_default_controls as analysis


HERE = Path(__file__).resolve().parent
CAMPAIGN = (
    HERE / "results" / "campaigns"
    / "default_controls_20260725T211840Z"
)


class AnalysisTests(unittest.TestCase):
    def test_recursive_campaign_manifest(self) -> None:
        analysis.verify_campaign_manifest(CAMPAIGN)

    def test_dual_replay_rebinds_all_requests(self) -> None:
        row = analysis.validate_dual(CAMPAIGN / "dual-warm-0", 0)
        self.assertEqual(row["request_count"], 74)
        self.assertEqual(row["slo_met_count"], 74)
        self.assertEqual(row["model_request_counts"], {
            "qwen3-14b-q4_k_m": 17,
            "qwen3-8b-q8_0": 57,
        })

    def test_swap_commands_are_stock_default(self) -> None:
        row = analysis.validate_stock_commands(CAMPAIGN / "swap-warm-0")
        self.assertEqual(row["load_count"], 10)
        self.assertEqual({item["slots"] for item in row["placements"]}, {4})

    def test_graph_writer_creates_assets(self) -> None:
        run = analysis.validate_dual(CAMPAIGN / "dual-warm-0", 0)
        run["label"] = "test"
        panel = analysis.bin_run(run)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            analysis.graph_svg([panel], path / "throughput.svg", False)
            analysis.graph_svg([panel], path / "energy.svg", True)
            for name in (
                    "throughput.svg", "throughput.png",
                    "energy.svg", "energy.png"):
                self.assertGreater((path / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
