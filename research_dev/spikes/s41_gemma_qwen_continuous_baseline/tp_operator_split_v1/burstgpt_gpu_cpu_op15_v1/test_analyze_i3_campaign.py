#!/usr/bin/env python3

import unittest

import analyze_i3_campaign


class CampaignAnalysisTest(unittest.TestCase):
    def test_percent(self):
        self.assertAlmostEqual(analyze_i3_campaign.percent(80.0, 100.0), -20.0)

    def test_positive_rejects_boolean(self):
        with self.assertRaises(analyze_i3_campaign.run_trace.RunError):
            analyze_i3_campaign.finite_positive(True, "test")

    def test_alternating_intervals(self):
        controls = [
            {"paid_start_ns": 10, "paid_end_ns": 20},
            {"paid_start_ns": 40, "paid_end_ns": 50},
            {"paid_start_ns": 70, "paid_end_ns": 80},
        ]
        treatments = [
            {"paid_start_ns": 21, "paid_end_ns": 30},
            {"paid_start_ns": 51, "paid_end_ns": 60},
            {"paid_start_ns": 81, "paid_end_ns": 90},
        ]
        self.assertTrue(
            analyze_i3_campaign.alternating_intervals(controls, treatments)
        )

    def test_rejects_nonalternating_intervals(self):
        controls = [
            {"paid_start_ns": 10, "paid_end_ns": 20},
            {"paid_start_ns": 25, "paid_end_ns": 35},
            {"paid_start_ns": 70, "paid_end_ns": 80},
        ]
        treatments = [
            {"paid_start_ns": 21, "paid_end_ns": 30},
            {"paid_start_ns": 51, "paid_end_ns": 60},
            {"paid_start_ns": 81, "paid_end_ns": 90},
        ]
        self.assertFalse(
            analyze_i3_campaign.alternating_intervals(controls, treatments)
        )


if __name__ == "__main__":
    unittest.main()
