#!/usr/bin/env python3

import copy
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import offload_solver as solver


class OffloadSolverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = solver.load_json(HERE / "current_profile.json")

    def load_example(self, name: str) -> dict:
        return solver.load_json(HERE / "examples" / name)

    def test_small_matmul_stays_on_desktop(self) -> None:
        workload = {
            "schema": solver.WORKLOAD_SCHEMA,
            "name": "small",
            "kind": "matmul",
            "host_backend": "cpu_isolated",
            "phone_backends": ["op15_htp"],
            "quantization": "q8_0",
            "activation_type": "f16",
            "m": 1,
            "n": 256,
            "k": 256,
        }
        result = solver.solve_workload(
            self.profile, workload, top=0
        )
        self.assertEqual(
            result["recommendation"]["decision"], "all_desktop"
        )
        self.assertGreater(
            result["baseline"]["median_ms"], 0.0
        )

    def test_matmul_finds_aligned_legal_split(self) -> None:
        workload = self.load_example("single_matmul_q8_cpu.json")
        result = solver.solve_workload(
            self.profile, workload, top=0
        )
        recommendation = result["recommendation"]
        self.assertEqual(recommendation["decision"], "offload")
        self.assertEqual(
            recommendation["phone_backend"], "op15_htp"
        )
        split = recommendation["split"]
        self.assertIn(split["axis"], {"n", "k", "all"})
        if split["axis"] == "n":
            self.assertEqual(split["phone_n"] % 32, 0)
            self.assertEqual(
                split["phone_n"] + split["host_n"], workload["n"]
            )
        if split["axis"] == "k":
            self.assertEqual(split["phone_k"] % 32, 0)
            self.assertEqual(
                split["phone_k"] + split["host_k"], workload["k"]
            )

    def test_moe_assigns_whole_experts(self) -> None:
        workload = self.load_example(
            "gemma4_26b_a4b_moe_q8_cpu.json"
        )
        result = solver.solve_workload(
            self.profile, workload, top=0
        )
        recommendation = result["recommendation"]
        self.assertEqual(recommendation["decision"], "offload")
        self.assertEqual(
            recommendation["strategy"], "whole_expert_parallel"
        )
        split = recommendation["split"]
        self.assertEqual(split["phone_active_experts"], 2)
        self.assertEqual(
            split["phone_active_experts"]
            + split["host_active_experts"],
            workload["active_experts"],
        )
        self.assertEqual(
            split["resident_experts"], workload["total_experts"]
        )

    def test_rtx_phone_candidates_require_staging_profile(self) -> None:
        workload = self.load_example("single_matmul_q8_cpu.json")
        workload = copy.deepcopy(workload)
        workload["host_backend"] = "rtx4060"
        result = solver.solve_workload(
            self.profile, workload, top=0
        )
        self.assertEqual(
            result["recommendation"]["decision"], "all_desktop"
        )
        self.assertEqual(result["candidate_count"]["eligible"], 0)
        blocker_text = " ".join(
            blocker
            for candidate in result[
                "diagnostic_blocked_candidates"
            ]
            for blocker in candidate["blockers"]
        )
        self.assertIn("D2H/H2D staging", blocker_text)

    def test_chain_keeps_phone_intermediates_inside_island(self) -> None:
        workload = self.load_example("ordered_chain_q8_cpu.json")
        result = solver.solve_workload(
            self.profile, workload, top=0
        )
        self.assertGreater(result["candidate_count"]["eligible"], 0)
        for candidate in result["candidates"]:
            for island in candidate["phone_islands"]:
                self.assertEqual(
                    island["transport"]["input_bytes"],
                    (
                        workload["operations"][island["start"]]["m"]
                        * workload["operations"][island["start"]]["k"]
                        * 2
                    ),
                )
                self.assertEqual(
                    island["transport"]["output_bytes"],
                    (
                        workload["operations"][island["end"] - 1][
                            "m"
                        ]
                        * workload["operations"][island["end"] - 1][
                            "n"
                        ]
                        * 2
                    ),
                )

    def test_chain_rejects_incompatible_shapes(self) -> None:
        workload = self.load_example("ordered_chain_q8_cpu.json")
        workload = copy.deepcopy(workload)
        workload["operations"][1]["k"] += 1
        with self.assertRaisesRegex(
            solver.SolverError, "does not feed"
        ):
            solver.solve_workload(self.profile, workload)

    def test_auto_transport_changes_at_large_payload(self) -> None:
        small = solver.estimate_transport(
            self.profile, 10 * 1024, 10 * 1024, "auto", 0.0
        )
        large = solver.estimate_transport(
            self.profile, 320 * 1024, 320 * 1024, "auto", 0.0
        )
        self.assertEqual(small["protocol"], "aoa")
        self.assertEqual(large["protocol"], "ncm")

    def test_unprofiled_batch_can_be_required_as_blocker(self) -> None:
        workload = {
            "schema": solver.WORKLOAD_SCHEMA,
            "name": "strict batch shape",
            "kind": "matmul",
            "host_backend": "cpu_isolated",
            "phone_backends": ["op15_htp"],
            "quantization": "q8_0",
            "activation_type": "f16",
            "m": 64,
            "n": 4096,
            "k": 4096,
            "require_profiled_shape": True,
            "split_axes": ["n", "k"],
        }
        result = solver.solve_workload(
            self.profile, workload, top=0
        )
        self.assertEqual(result["candidate_count"]["eligible"], 0)
        blockers = " ".join(
            blocker
            for candidate in result[
                "diagnostic_blocked_candidates"
            ]
            for blocker in candidate["blockers"]
        )
        self.assertIn("outside the profiled batch regime", blockers)

    def test_json_result_is_ascii_serializable(self) -> None:
        workload = self.load_example("gemma4_12b_ffn_q8_cpu.json")
        result = solver.solve_workload(
            self.profile, workload, top=3
        )
        json.dumps(solver.rounded(result), sort_keys=True).encode(
            "ascii"
        )


if __name__ == "__main__":
    unittest.main()
