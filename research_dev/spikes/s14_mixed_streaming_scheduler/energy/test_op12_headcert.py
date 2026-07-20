#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest

import op12_headcert as O


def eligible_row() -> dict:
    tokens = [1, 2, 3]
    return {
        "placement_status": "SCHEDULED_PLACEMENT_OK",
        "missing_buffer_compute_nodes": 0,
        "host_returncode": 0,
        "n_requests_measured": 8,
        "token_ids": tokens,
        "token_ids_by_request": [tokens[:] for _ in range(8)],
        "placement_cert": {
            "compute_by_op_and_buffer": {
                "MUL_MAT": {"HTP0": 10},
                "GET_ROWS": {"CPU": 1},
            },
        },
    }


def thermal(value: int = 40_000) -> dict:
    return {
        "valid": True,
        "sensors_millic": {"nsphmx-0": value},
        "max_millic": value,
    }


class OP12PointTests(unittest.TestCase):
    def test_accepts_exact_complete_point(self) -> None:
        self.assertTrue(O.point_passes(eligible_row(), [1, 2, 3], 8, thermal(), thermal()))

    def test_rejects_missing_request_tokens(self) -> None:
        row = eligible_row()
        row["token_ids_by_request"] = row["token_ids_by_request"][:-1]
        self.assertFalse(O.point_passes(row, [1, 2, 3], 8, thermal(), thermal()))

    def test_rejects_late_request_mismatch(self) -> None:
        row = eligible_row()
        row["token_ids_by_request"][-1] = [9]
        self.assertFalse(O.point_passes(row, [1, 2, 3], 8, thermal(), thermal()))

    def test_rejects_undeclared_cpu_compute(self) -> None:
        row = copy.deepcopy(eligible_row())
        row["placement_cert"]["compute_by_op_and_buffer"]["MUL_MAT"] = {"CPU": 10}
        self.assertFalse(O.point_passes(row, [1, 2, 3], 8, thermal(), thermal()))

    def test_rejects_hot_end(self) -> None:
        self.assertFalse(O.point_passes(eligible_row(), [1, 2, 3], 8, thermal(), thermal(86_000)))

    def test_rejects_bool_count(self) -> None:
        row = copy.deepcopy(eligible_row())
        row["placement_cert"]["compute_by_op_and_buffer"]["MUL_MAT"] = {"HTP0": True}
        self.assertFalse(O.point_passes(row, [1, 2, 3], 8, thermal(), thermal()))


if __name__ == "__main__":
    unittest.main()
