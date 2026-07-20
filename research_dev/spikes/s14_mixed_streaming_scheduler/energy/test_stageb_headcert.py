#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest

import stageb_headcert as SB


def eligible_row() -> dict:
    return {
        "placement_status": "SCHEDULED_PLACEMENT_OK",
        "missing_buffer_compute_nodes": 0,
        "host_returncode": 0,
        "n_requests_measured": 8,
        "token_match_vs_mono": True,
        "all_tokens_match_vs_mono": True,
        "placement_cert": {
            "compute_by_op_and_buffer": {
                "MUL_MAT": {"HTP0": 4},
                "GET_ROWS": {"CPU": 1},
            },
        },
    }


class StageBEligibilityTests(unittest.TestCase):
    def test_thermal_gate_recomputes_maximum(self) -> None:
        snapshot = {
            "valid": True,
            "sensors_millic": {"nsphmx-0": 40_000, "nsphmx-1": 41_000},
            "max_millic": 41_000,
        }
        self.assertTrue(SB.thermal_ok(snapshot, 60_000))
        snapshot["max_millic"] = 40_000
        self.assertFalse(SB.thermal_ok(snapshot, 60_000))

    def test_accepts_htp_with_declared_get_rows(self) -> None:
        self.assertTrue(SB.row_is_eligible(eligible_row(), 8))

    def test_rejects_token_mismatch(self) -> None:
        row = eligible_row()
        row["token_match_vs_mono"] = False
        self.assertFalse(SB.row_is_eligible(row, 8))

    def test_rejects_nonfirst_request_mismatch(self) -> None:
        row = eligible_row()
        row["all_tokens_match_vs_mono"] = False
        self.assertFalse(SB.row_is_eligible(row, 8))

    def test_rejects_incomplete_request_set(self) -> None:
        row = eligible_row()
        row["n_requests_measured"] = 7
        self.assertFalse(SB.row_is_eligible(row, 8))

    def test_rejects_undeclared_cpu_compute(self) -> None:
        row = copy.deepcopy(eligible_row())
        row["placement_cert"]["compute_by_op_and_buffer"]["MUL_MAT"] = {"CPU": 4}
        self.assertFalse(SB.row_is_eligible(row, 8))

    def test_rejects_zero_htp_compute(self) -> None:
        row = eligible_row()
        row["placement_cert"]["compute_by_op_and_buffer"] = {"GET_ROWS": {"CPU": 1}}
        self.assertFalse(SB.row_is_eligible(row, 8))


if __name__ == "__main__":
    unittest.main()
