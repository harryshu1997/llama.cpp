#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
for path in (REPO_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multi_session_phone_v1.build_op15_residency_plan import (  # noqa: E402
    MIB,
    build_plan,
)
from multi_session_phone_v1.analyze_qwen_full_energy_pair import (  # noqa: E402
    canonical_bytes,
)
from multi_session_phone_v1.plan_unified_qwen_screen import (  # noqa: E402
    compile_decision,
)
from research_dev.scheduler import PhoneResidencyPlan  # noqa: E402


EXPERIMENT = ROOT / "multi_session_phone_v1"
RESULT = EXPERIMENT / "results/QWEN_FULL_FFN_ENERGY_SCREEN_R1_R2.json"
M1_M4_RESULT = (
    EXPERIMENT
    / "results/QWEN_FULL_FFN_M1_M4_ENERGY_SCREEN_ABBA_V2.json"
)
PLAN = EXPERIMENT / "results/OP15_THREE_SESSION_RESIDENCY_PLAN_V1.json"
DECISION = EXPERIMENT / "results/UNIFIED_QWEN_FULL_FFN_DECISION_V1.json"
QWEN_ARM = EXPERIMENT / "run_qwen_full_energy_arm.sh"


class MultiSessionPhoneTests(unittest.TestCase):
    def test_rebalanced_layout_selects_matching_phone_binaries(self) -> None:
        source = QWEN_ARM.read_text(encoding="ascii")
        self.assertIn(
            "default_phone_workers=$phone_base/"
            "resident_ffn_workers-rebalance-v1",
            source,
        )
        self.assertIn(
            "default_phone_session=$phone_base/"
            "resident_ffn_session-rebalance-v1.sh",
            source,
        )

    def test_optional_llama_slice_uses_a_fourth_htp_session(self) -> None:
        workers = (EXPERIMENT / "resident_ffn_workers.cpp").read_text(
            encoding="ascii"
        )
        session = (EXPERIMENT / "resident_ffn_session.sh").read_text(
            encoding="ascii"
        )
        self.assertIn('"resident-llama-ffn", argv[3], "HTP3"', workers)
        self.assertIn('llama_layers, "8192"', workers)
        self.assertIn('"0.0.0.0", llama_port', workers)
        self.assertIn("GGML_HEXAGON_NDEV=4", session)
        self.assertIn("S42_LLAMA_FFN_MODEL", session)
        worker = (
            REPO_ROOT / "examples/layersplit/ffn-split-worker.cpp"
        ).read_text(encoding="ascii")
        self.assertIn('strcmp(architecture, "llama") == 0', worker)

    def test_residency_plan_uses_three_mapping_arenas_and_one_htp(self) -> None:
        plan = build_plan()
        self.assertEqual(plan.resident_bytes, 9_673_170_944)
        self.assertEqual(
            [session.compute_backend for session in plan.sessions],
            ["HTP0", "HTP1", "HTP2"],
        )
        self.assertEqual(plan.shared_compute_resource_id, "op15-htp")
        self.assertGreaterEqual(plan.minimum_available_bytes, 2 * 1024 * MIB)
        self.assertEqual(
            [row.physical_m_max for row in plan.sessions[1].slices], [4]
        )
        self.assertEqual(
            [row.physical_m_max for row in plan.sessions[2].slices], [4]
        )

    def test_checked_in_plan_matches_public_scheduler_builder(self) -> None:
        value = json.loads(PLAN.read_text(encoding="ascii"))
        parsed = PhoneResidencyPlan.from_json(value)
        self.assertEqual(parsed, build_plan())

    def test_repeated_energy_screen_passes_all_gates(self) -> None:
        value = json.loads(RESULT.read_text(encoding="ascii"))
        supplied_hash = value.pop("record_sha256")
        self.assertEqual(
            supplied_hash,
            hashlib.sha256(canonical_bytes(value)).hexdigest(),
        )
        self.assertEqual(value["status"], "PASS")
        self.assertTrue(all(value["gates"].values()))
        self.assertAlmostEqual(
            value["comparison"]["changes"]["fleet_j_change_pct"],
            -39.32788261808692,
        )
        self.assertEqual(
            value["comparison"]["pair_fleet_energy_savings_pct"],
            [39.005731824631184, 39.647425086564645],
        )

    def test_m1_m4_energy_screen_passes_all_gates(self) -> None:
        value = json.loads(M1_M4_RESULT.read_text(encoding="ascii"))
        supplied_hash = value.pop("record_sha256")
        self.assertEqual(
            supplied_hash,
            hashlib.sha256(canonical_bytes(value)).hexdigest(),
        )
        self.assertEqual(value["status"], "PASS")
        self.assertTrue(all(value["gates"].values()))
        self.assertFalse(value["observations"]["control_repeats_exact_tokens"])
        self.assertAlmostEqual(
            value["comparison"]["changes"]["fleet_j_change_pct"],
            -30.496340260003386,
        )

    def test_unified_scheduler_selects_the_composite_qwen_arm(self) -> None:
        result = json.loads(RESULT.read_text(encoding="ascii"))
        plan = PhoneResidencyPlan.from_json(json.loads(
            PLAN.read_text(encoding="ascii")))
        decision = compile_decision(result, plan)
        checked_in = json.loads(DECISION.read_text(encoding="ascii"))
        self.assertEqual(decision, checked_in)
        self.assertEqual(
            [signal["session_id"] for signal in decision["arm"]["signals"]],
            ["htp1", "htp2"],
        )
        self.assertEqual(
            {lease["resource_id"] for lease in decision["leases"]},
            {"op15-htp", "op15-functionfs", "desktop-usb-root"},
        )


if __name__ == "__main__":
    unittest.main()
