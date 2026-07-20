#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path

import cp_d_priority as cpd


HERE = Path(__file__).resolve().parent


class CPDTests(unittest.TestCase):
    def test_measured_profiles_drive_knee_and_max_batch(self) -> None:
        bge, knee, exact, _ = cpd.load_bge_profile(cpd.BGE_PROFILE, 32)
        gemma, _ = cpd.load_gemma_profile(HERE / "stage_d_batch_scaling.json", 16)
        decisions = cpd.policy_decisions(bge, gemma)
        self.assertEqual((knee, exact), (16, 31))
        self.assertEqual(decisions["high"]["batch"], 16)
        self.assertEqual(decisions["low"]["batch"], 16)
        self.assertTrue(decisions["compatibility_isolation_enforced"])

    def test_contradictory_batch_two_is_excluded(self) -> None:
        gemma, _ = cpd.load_gemma_profile(HERE / "stage_d_batch_scaling.json", 16)
        self.assertEqual([point.batch_size for point in gemma], [1, 4, 8, 16])

    def test_energy_window_repetitions_are_bounded_from_below(self) -> None:
        self.assertEqual(cpd.required_reps(4_000, 30, 5.0), 1250)
        self.assertEqual(cpd.required_reps(4_000, 2000, 5.0), 2000)
        with self.assertRaises(cpd.CPDError):
            cpd.required_reps(0, 30, 5.0)

    def test_placement_certificate_gate(self) -> None:
        good = (
            'PLACEMENTCERT {"compute_nodes":10,"missing_buffer_compute_nodes":0,'
            '"by_buffer":{"CUDA0":10},"non_htp_ops":[],"status":"SCHEDULED_PLACEMENT_OK"}'
        )
        self.assertEqual(cpd.parse_placement(good)["compute_nodes"], 10)
        with self.assertRaises(cpd.CPDError):
            cpd.parse_placement(good.replace('"CUDA0":10', '"CPU":10'))

    def test_persisted_power_window_is_integer_and_replayable(self) -> None:
        class Sampler:
            samples = [
                (0.9, 10.0, 50.0, "P0"),
                (1.0, 20.0, 60.0, "P0"),
                (1.1, 30.0, 70.0, "P0"),
                (1.2, 40.0, 80.0, "P0"),
                (1.3, 50.0, 90.0, "P0"),
                (1.4, 60.0, 100.0, "P0"),
                (1.5, 70.0, 100.0, "P0"),
                (1.6, 80.0, 100.0, "P0"),
            ]

        result = cpd.persisted_energy_window(Sampler(), 1.05, 1.55)
        self.assertEqual(result["energy_nj"], 22_500_000_000)
        self.assertEqual(result["paid_window_us"], {"start": 1_050_000, "end": 1_550_000})
        self.assertEqual(result["n_samples"], 5)


if __name__ == "__main__":
    unittest.main()
