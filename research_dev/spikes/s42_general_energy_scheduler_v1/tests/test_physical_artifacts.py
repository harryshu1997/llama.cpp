#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_replay import analyze, load_replay  # noqa: E402
from physical_iteration import (  # noqa: E402
    IterationError,
    load_energy_receipt,
    validate_iteration_record,
)
from replay import load_profile  # noqa: E402
from render_physical_iteration import render  # noqa: E402


PROFILE = ROOT / "physical_4060ti_op15_profile.json"
REPLAY = ROOT / "BURSTGPT_REPLAY_V1.json"
LIVE_REPLAY = ROOT / "BURSTGPT_LIVE_READINESS_DRY_RUN_V1.json"


class PhysicalArtifactTests(unittest.TestCase):
    def test_physical_profile_hash_and_routes(self) -> None:
        _, profile = load_profile(PROFILE)
        self.assertEqual(len(profile.routes), 3)
        self.assertEqual(profile.resources["cuda0"].capacity, 8)

    def test_replay_hash_and_route_counts(self) -> None:
        replay = load_replay(REPLAY)
        modes = {row["mode"]: row for row in replay["modes"]}
        self.assertEqual(modes["enforce"]["route_counts"]["cold-cpu-task"], 17)
        self.assertEqual(
            modes["shadow"]["route_counts"]["cold-cpu-op15-ffn"], 17
        )

    def test_analysis_reproduces_passing_comparison(self) -> None:
        result = analyze(PROFILE, REPLAY)
        self.assertTrue(result["pass"])
        self.assertEqual(result["enforce_phone_request_count"], 0)

    def test_live_readiness_overlay_fails_closed(self) -> None:
        replay = load_replay(LIVE_REPLAY)
        for mode in replay["modes"]:
            self.assertNotIn("cold-cpu-op15-ffn", mode["route_counts"])
            self.assertEqual(
                mode["rejection_counts"]["resource is not ready: op15-htp"],
                17,
            )

    def test_exact_quality_replay_rejects_approximate_route(self) -> None:
        path = ROOT / "BURSTGPT_REPLAY_EXACT_V1.json"
        replay = load_replay(path)
        for mode in replay["modes"]:
            self.assertEqual(mode["rejection_counts"]["QUALITY_INSUFFICIENT"], 17)

    def test_live_probe_records_concurrent_process_ownership(self) -> None:
        value = json.loads(
            (ROOT / "LIVE_PROBE_4060TI_OP15_V1.json").read_text(encoding="ascii")
        )
        self.assertTrue(value["identity_ready"])
        self.assertFalse(value["execution_route_ready"])
        self.assertIsInstance(value["active_inference_processes"], list)

    def test_physical_iteration_has_clear_real_device_headline(self) -> None:
        value = json.loads(
            (ROOT / "PHYSICAL_ITERATION_I0.json").read_text(encoding="ascii")
        )
        record = validate_iteration_record(value)
        headline = record["real_device_headline"]
        self.assertAlmostEqual(
            headline["latency"]["control_average_s"], 327.704644393
        )
        self.assertAlmostEqual(
            headline["latency"]["treatment_average_s"], 197.32124562
        )
        self.assertEqual(headline["phone_work"]["paid_phone_calls"], 24240)
        self.assertAlmostEqual(
            headline["phone_work"]["phone_fraction_eligible_cold_ffn_macs"],
            0.6261975193315689,
        )
        self.assertIsNone(headline["accounted_fleet_energy"]["control"])
        self.assertEqual(
            record["verdict"], "INCOMPLETE_MISSING_ACCOUNTED_DEVICE_ENERGY"
        )
        overlap = headline["phone_work"]["overlap_p50_diagnostic"]
        self.assertAlmostEqual(
            overlap["exposed_join_wait_fraction"], 0.046722, places=5
        )
        self.assertEqual(
            headline["phone_work"]["overlap_mean"]["status"],
            "MISSING_SUM_COUNT_COUNTERS",
        )

    def test_physical_iteration_renderer_keeps_missing_energy_visible(self) -> None:
        value = json.loads(
            (ROOT / "PHYSICAL_ITERATION_I0.json").read_text(encoding="ascii")
        )
        text = render(value)
        self.assertIn("327.704644 s", text)
        self.assertIn("197.321246 s", text)
        self.assertIn(
            "| accounted fleet energy | MISSING | MISSING | MISSING |", text
        )
        self.assertIn("p50 exposed join wait: 4.67%", text)
        self.assertIn("phone_work_pct=62.62", text)

    @staticmethod
    def energy_receipt() -> dict[str, object]:
        duration_s = 10.0
        components = [
            {
                "kind": "server_cpu_package",
                "device_id": "cpu-0",
                "source": "rapl",
                "sample_count": 10,
                "average_power_w": 50.0,
                "duration_s": duration_s,
                "energy_j": 500.0,
            },
            {
                "kind": "server_gpu_board",
                "device_id": "gpu-0",
                "source": "nvml",
                "sample_count": 10,
                "average_power_w": 100.0,
                "duration_s": duration_s,
                "energy_j": 1000.0,
            },
            {
                "kind": "phone_system",
                "device_id": "phone-0",
                "source": "usb-meter",
                "sample_count": 10,
                "average_power_w": 4.0,
                "duration_s": duration_s,
                "energy_j": 40.0,
            },
        ]
        return {
            "schema": "s42-accounted-device-energy-receipt-v1",
            "valid": True,
            "boundary_scope": "accounted_devices",
            "boundary_id": "components-v1",
            "result_sha256": "sha256:result",
            "paid_start_ns": 0,
            "paid_end_ns": 10_000_000_000,
            "components": components,
            "excluded_components": ["dram", "motherboard"],
            "energy_j": 1540.0,
        }

    def write_energy_receipt(self, root: Path, value: object) -> Path:
        path = root / "ENERGY.json"
        path.write_text(json.dumps(value), encoding="ascii")
        return path

    def test_component_energy_receipt_reconciles_power_times_time(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = self.write_energy_receipt(Path(raw), self.energy_receipt())
            receipt = load_energy_receipt(
                path,
                {
                    "sha256": "sha256:result",
                    "makespan_s": 10.0,
                    "paid_start_ns": 0,
                    "paid_end_ns": 10_000_000_000,
                },
            )
        self.assertEqual(receipt["energy_j"], 1540.0)

    def test_component_energy_receipt_requires_phone_in_control(self) -> None:
        value = self.energy_receipt()
        value["components"] = value["components"][:-1]  # type: ignore[index]
        value["energy_j"] = 1500.0
        with tempfile.TemporaryDirectory() as raw:
            path = self.write_energy_receipt(Path(raw), value)
            with self.assertRaises(IterationError):
                load_energy_receipt(
                    path,
                    {
                        "sha256": "sha256:result",
                        "makespan_s": 10.0,
                        "paid_start_ns": 0,
                        "paid_end_ns": 10_000_000_000,
                    },
                )

    def test_component_energy_receipt_rejects_unreconciled_total(self) -> None:
        value = self.energy_receipt()
        value["energy_j"] = 1500.0
        with tempfile.TemporaryDirectory() as raw:
            path = self.write_energy_receipt(Path(raw), value)
            with self.assertRaises(IterationError):
                load_energy_receipt(
                    path,
                    {
                        "sha256": "sha256:result",
                        "makespan_s": 10.0,
                        "paid_start_ns": 0,
                        "paid_end_ns": 10_000_000_000,
                    },
                )

    def test_component_energy_receipt_must_match_paid_interval(self) -> None:
        value = self.energy_receipt()
        value["paid_start_ns"] = 1
        value["paid_end_ns"] = 10_000_000_001
        with tempfile.TemporaryDirectory() as raw:
            path = self.write_energy_receipt(Path(raw), value)
            with self.assertRaises(IterationError):
                load_energy_receipt(
                    path,
                    {
                        "sha256": "sha256:result",
                        "makespan_s": 10.0,
                        "paid_start_ns": 0,
                        "paid_end_ns": 10_000_000_000,
                    },
                )


if __name__ == "__main__":
    unittest.main()
