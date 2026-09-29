#!/usr/bin/env python3

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_AB = ROOT / "physical_ab_v1"
if str(PHYSICAL_AB) not in sys.path:
    sys.path.insert(0, str(PHYSICAL_AB))

from analyze_stage6_arm import analyze  # noqa: E402
from analyze_stage6_pair import (  # noqa: E402
    PairError,
    validate_scheduler_binding,
)
from render_stage6_pair import canonical_bytes, render  # noqa: E402


class Stage6PhysicalABTests(unittest.TestCase):
    def test_checked_in_pair_and_gate_receipts_are_bound(self) -> None:
        pair = json.loads(
            (PHYSICAL_AB / "STAGE6_PHYSICAL_AB_4060TI_OP15_V1.json").read_text(
                encoding="ascii"
            )
        )
        supplied = pair.pop("record_sha256")
        self.assertEqual(supplied, hashlib.sha256(canonical_bytes(pair)).hexdigest())
        pair["record_sha256"] = supplied
        self.assertEqual(pair["status"], "PASS")
        self.assertAlmostEqual(
            pair["comparison"]["makespan_change_pct"], -13.829343, places=5
        )
        self.assertIn("631.293 s", render(pair))

        boundary = json.loads(
            (PHYSICAL_AB / "STAGE6_TREATMENT_RUNTIME_GATE_BOUNDARY_V1.json")
            .read_text(encoding="ascii")
        )
        treatment = json.loads(
            (PHYSICAL_AB / "STAGE6_TREATMENT_RUNTIME_GATE_V2.json").read_text(
                encoding="ascii"
            )
        )
        self.assertEqual(boundary["status"], "FAIL")
        self.assertEqual(boundary["transport"]["last_observed_calls"], 52288)
        self.assertEqual(treatment["status"], "PASS")
        self.assertEqual(treatment["transport"]["last_observed_calls"], 52320)

    def test_checked_in_unified_scheduler_pair_is_bound(self) -> None:
        pair = json.loads(
            (
                PHYSICAL_AB
                / "UNIFIED_SCHEDULER_BURSTGPT_4060TI_OP15_R5_V1.json"
            ).read_text(encoding="ascii")
        )
        supplied = pair.pop("record_sha256")
        self.assertEqual(supplied, hashlib.sha256(canonical_bytes(pair)).hexdigest())
        self.assertEqual(pair["status"], "PASS")
        self.assertAlmostEqual(
            pair["comparison"]["fleet_energy_change_pct"],
            -16.000981,
            places=5,
        )
        self.assertEqual(
            pair["scheduler_plans"]["treatment"]["route_id"],
            "i3-cold-cpu-op15-ffn-v1",
        )
        self.assertEqual(
            pair["scheduler_plans"]["treatment"]["decision_reason"],
            "VERIFIED_COHORT_ENERGY_SAVING",
        )

    def write_case(
        self,
        root: Path,
        *,
        mode: str = "op15",
        phone_status: int = 0,
        heartbeat_gap_ns: int = 500_000_000,
    ) -> tuple[Path, Path, Path]:
        result = {
            "mode": mode,
            "paid_end_ns": 900_000_000,
            "paid_start_ns": 100_000_000,
            "phone": {
                "bridge": (
                    {"calls": 52320, "reset_recoveries": 0}
                    if mode == "op15" else None
                ),
            },
            "preflight": {
                "cold_model_sha256": (
                    "494518c2262a26e2a607af0e40bca11c4de5a0b108e7c21308906dbbfb1c6f8c"
                ),
                "hot_model_sha256": (
                    "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"
                ),
            },
            "repeat_index": 1,
            "schema": "s41-burstgpt-llama-server-result-v1",
            "status": "PASS",
        }
        result_path = root / "RESULT.json"
        result_path.write_text(json.dumps(result), encoding="ascii")
        host_rows = [{
            "cpu_package_throttle_count": 7,
            "gpu": {
                "hardware_thermal_slowdown": "Not Active",
                "software_thermal_slowdown": "Not Active",
                "temperature_c": 62,
                "uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            },
            "kind": "telemetry",
            "monotonic_ns": 500_000_000,
            "processes": {
                "bridge": [3] if mode == "op15" else [],
                "cold": [2],
                "hot": [1],
            },
            "usb": [{
                "speed_mbps": 5000,
                "vendor_product": "18d1:2d00" if mode == "op15" else "22d9:2772",
            }],
        }]
        if mode == "op15":
            host_rows.extend([
                {
                    "calls": 32,
                    "file_mtime_ns": 1_000_000_000,
                    "kind": "bridge_progress",
                    "monotonic_ns": 200_000_000,
                    "reset_recoveries": 0,
                },
                {
                    "calls": 52320,
                    "file_mtime_ns": 1_000_000_000 + heartbeat_gap_ns,
                    "kind": "bridge_progress",
                    "monotonic_ns": 800_000_000,
                    "reset_recoveries": 0,
                },
            ])
        host_path = root / "host.jsonl"
        host_path.write_text(
            "".join(json.dumps(row) + "\n" for row in host_rows),
            encoding="ascii",
        )
        phone_path = root / "phone.tsv"
        phone_path.write_text(
            "epoch_s\tuptime_s\tthermal_status\tbattery_millic\t"
            "shell_millic\tcpu_millic\tnpu_millic\tgpu_millic\tddr_millic\n"
            f"1\t2.0\t{phone_status}\t30000\t31000\t40000\t42000\t39000\t35000\n",
            encoding="ascii",
        )
        return result_path, host_path, phone_path

    def test_treatment_passes_complete_runtime_gates(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            paths = self.write_case(Path(raw))
            result = analyze("op15", 1, *paths, 1.0)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["transport"]["last_observed_calls"], 52320)

    def test_control_does_not_require_bridge_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            paths = self.write_case(Path(raw), mode="cpu")
            result = analyze("cpu", 1, *paths, 1.0)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["gates"]["worker_transport_heartbeat"])

    def test_phone_thermal_status_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            paths = self.write_case(Path(raw), phone_status=2)
            result = analyze("op15", 1, *paths, 1.0)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["gates"]["op15_android_thermal_status_none"])

    def test_idle_gap_between_rpc_bursts_is_not_worker_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            paths = self.write_case(
                Path(raw), heartbeat_gap_ns=1_500_000_000
            )
            result = analyze("op15", 1, *paths, 1.1)
        self.assertEqual(result["status"], "PASS")
        self.assertGreater(result["transport"]["progress_max_gap_s"], 1.0)

    def test_pair_accepts_bound_unified_scheduler_plan(self) -> None:
        gate = {
            "gates": {"unified_scheduler_plan_bound": True},
            "scheduler": {
                "decision_reason": "VERIFIED_COHORT_ENERGY_SAVING",
                "plan_sha256": "sha256:" + "a" * 64,
                "route_id": "cpu-op15-ffn",
            },
        }
        self.assertEqual(
            validate_scheduler_binding(gate, Path("receipt.json")),
            gate["scheduler"],
        )

    def test_pair_rejects_unverified_scheduler_plan(self) -> None:
        gate = {
            "gates": {"unified_scheduler_plan_bound": False},
            "scheduler": {
                "decision_reason": "VERIFIED_COHORT_ENERGY_SAVING",
                "plan_sha256": "sha256:" + "a" * 64,
                "route_id": "cpu-op15-ffn",
            },
        }
        with self.assertRaises(PairError):
            validate_scheduler_binding(gate, Path("receipt.json"))


if __name__ == "__main__":
    unittest.main()
