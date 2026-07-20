#!/usr/bin/env python3

import math
import unittest
from array import array

import run_cp1_middle as gate


class Cp1MiddleTests(unittest.TestCase):
    def test_repo_root_is_bound_to_this_checkout(self) -> None:
        self.assertTrue((gate.REPO / "AGENTS.md").is_file())

    def test_rel_l2_identity(self) -> None:
        values = array("f", [1.0, 2.0, 3.0])
        self.assertEqual(gate.rel_l2(values, values), 0.0)

    def test_rel_l2_rejects_nonfinite(self) -> None:
        self.assertTrue(math.isinf(gate.rel_l2(array("f", [math.nan]), array("f", [1.0]))))

    def test_json_safe_replaces_nonfinite(self) -> None:
        self.assertEqual(gate.json_safe({"x": [math.inf, 1.0]}), {"x": [None, 1.0]})

    def test_argmax_is_row_local(self) -> None:
        old = gate.N_EMBD
        try:
            gate.N_EMBD = 3
            lhs = array("f", [3.0, 2.0, 1.0, 1.0, 4.0, 2.0])
            rhs = array("f", [3.0, 2.0, 1.0, 5.0, 4.0, 2.0])
            self.assertEqual(gate.row_argmax_mismatches(lhs, rhs, 2), 1)
        finally:
            gate.N_EMBD = old

    def test_placement_accepts_declared_cpu_get_rows(self) -> None:
        cert = {
            "status": "SCHEDULED_PLACEMENT_OK",
            "layer_start": 6,
            "layer_end": 12,
            "missing_buffer_compute_nodes": 0,
            "compute_by_op_and_buffer": {
                "MUL_MAT": {"HTP0": 4},
                "GET_ROWS": {"CPU": 1},
            },
        }
        self.assertEqual(gate.placement_problems(cert, "HTP0"), [])

    def test_placement_rejects_undeclared_cpu(self) -> None:
        cert = {
            "status": "SCHEDULED_PLACEMENT_OK",
            "layer_start": 6,
            "layer_end": 12,
            "missing_buffer_compute_nodes": 0,
            "compute_by_op_and_buffer": {"MUL_MAT": {"CPU": 4}},
        }
        problems = gate.placement_problems(cert, "HTP0")
        self.assertIn("undeclared placement MUL_MAT@CPU", problems)
        self.assertIn("no HTP0 compute", problems)


if __name__ == "__main__":
    unittest.main()
