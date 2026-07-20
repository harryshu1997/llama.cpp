#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest
from fractions import Fraction


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("analyze", HERE / "analyze.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def fixture():
    return {
        "schema": "s11-fixed-route-poc-result-v2",
        "treatment_route": "A0_OP15",
        "aggregate": {"exact_work_all_pairs": True},
        "pairs": [{
            "pair_index": 0,
            "exact_work": True,
            "treatment_route": "A0_OP15",
            "control_batch_metrics": {
                "batch_size": 8,
                "group_records": [{
                    "batch_index": 0,
                    "request_wall_us": 204_764,
                }],
            },
            "treatment_batch_metrics": {
                "batch_size": 8,
                "group_records": [{
                    "batch_index": 0,
                    "request_wall_us": 449_656,
                    "host_us": 202_954,
                    "stage_a_us": 246_687,
                    "stage_b_us": 0,
                    "prefill_us": 191_040,
                    "decode_us": 258_601,
                }],
            },
        }],
    }


class FeasibilityTests(unittest.TestCase):
    def test_current_b8_bound(self):
        result = MOD.analyze_summary(fixture())
        row = result["rows"][0]
        self.assertEqual(row["treatment_gap_us"], 246_702)
        self.assertEqual(row["treatment_phone_us"], 246_687)
        self.assertEqual(row["treatment_residual_us"], 15)
        self.assertEqual(
            row["tail_time_vs_control"]["decimal"], "0.991160556")
        self.assertFalse(row["equal_tail_power_has_nonnegative_solution"])
        self.assertEqual(
            row["sensitivity"][0]["max_tail_power_ratio"]["decimal"],
            "0.908026449")

    def test_threshold_closes_exactly(self):
        row = MOD.extract_timing_rows(fixture())[0]
        target = Fraction(9, 10)
        gap_ratio = Fraction(1, 10)
        analyzed = MOD.analyze_row(row, target, [gap_ratio])
        tail_ratio_record = analyzed["sensitivity"][0]["max_tail_power_ratio"]
        tail_ratio = Fraction(
            tail_ratio_record["numerator"], tail_ratio_record["denominator"])
        ratio = (
            tail_ratio * row["treatment_tail_us"] +
            gap_ratio * row["treatment_gap_us"]
        ) / row["control_wall_us"]
        self.assertEqual(ratio, target)

    def test_rejects_inexact_or_invalid_timing(self):
        data = fixture()
        data["pairs"][0]["exact_work"] = False
        with self.assertRaises(MOD.AnalysisError):
            MOD.analyze_summary(data)

        data = fixture()
        data["pairs"][0]["treatment_batch_metrics"]["group_records"][0][
            "host_us"] = 500_000
        with self.assertRaises(MOD.AnalysisError):
            MOD.analyze_summary(data)

    def test_rejects_bad_sensitivity_input(self):
        with self.assertRaises(MOD.AnalysisError):
            MOD.analyze_summary(fixture(), gap_power_bps=[False])

    def test_matches_groups_by_batch_index(self):
        data = fixture()
        control = data["pairs"][0]["control_batch_metrics"]["group_records"]
        treatment = data["pairs"][0]["treatment_batch_metrics"]["group_records"]
        control.append({"batch_index": 1, "request_wall_us": 300_000})
        treatment.append({
            "batch_index": 1,
            "request_wall_us": 600_000,
            "host_us": 250_000,
            "stage_a_us": 349_900,
            "stage_b_us": 0,
            "prefill_us": 200_000,
            "decode_us": 399_900,
        })
        treatment.reverse()
        rows = MOD.extract_timing_rows(data)
        self.assertEqual([row["group_index"] for row in rows], [0, 1])
        self.assertEqual(rows[0]["control_wall_us"], 204_764)
        self.assertEqual(rows[1]["control_wall_us"], 300_000)

        data["pairs"][0]["treatment_batch_metrics"]["group_records"][0][
            "batch_index"] = 2
        with self.assertRaises(MOD.AnalysisError):
            MOD.extract_timing_rows(data)

    def test_rejects_wrong_summary_identity(self):
        mutations = [
            ("schema", "wrong"),
            ("treatment_route", "A0_OP15_OP12"),
        ]
        for field, value in mutations:
            data = fixture()
            data[field] = value
            with self.subTest(field=field):
                with self.assertRaises(MOD.AnalysisError):
                    MOD.extract_timing_rows(data)

        data = fixture()
        data["aggregate"]["exact_work_all_pairs"] = False
        with self.assertRaises(MOD.AnalysisError):
            MOD.extract_timing_rows(data)

        data = fixture()
        data["pairs"].append(dict(data["pairs"][0]))
        with self.assertRaises(MOD.AnalysisError):
            MOD.extract_timing_rows(data)


if __name__ == "__main__":
    unittest.main()
