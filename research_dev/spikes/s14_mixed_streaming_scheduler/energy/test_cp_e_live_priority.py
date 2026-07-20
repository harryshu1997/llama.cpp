#!/usr/bin/env python3

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import cp_e_live_priority as CPE


def row(label: str, repeat: int, energy: float, high_p95: float, phone_live: bool) -> dict:
    return {
        "label": label,
        "repeat": repeat,
        "bge": {"encodes": 256, "lat_us_p95": high_p95},
        "gemma": {"generated_tokens": 128, "phone_live": phone_live, "lat_us_p95": 100.0},
        "concurrent_overlap_s": 5.0,
        "concurrent_overlap_fraction_of_shorter": 1.0,
        "selected_gpu_cohort": {"energy_j": energy},
    }


class LivePriorityTests(unittest.TestCase):
    def test_failure_logs_are_persisted(self) -> None:
        high = CPE.ReadyProcess([], {}, "READY", None)
        low = CPE.ReadyProcess([], {}, "READY", "DONE")
        high.stdout_lines = ["high-out"]
        high.stderr_lines = ["high-error"]
        low.stdout_lines = ["low-out"]
        low.stderr_lines = ["low-error"]
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            CPE.persist_process_logs(log_dir, high, low)
            self.assertEqual((log_dir / "high.stderr").read_text(), "high-error")
            self.assertEqual((log_dir / "low.stderr").read_text(), "low-error")

    def test_host_placement_gate(self) -> None:
        cert = {
            "status": "SCHEDULED_PLACEMENT_OK",
            "missing_buffer_compute_nodes": 0,
            "layer_start": 12,
            "layer_end": 48,
            "compute_by_buffer_type": {"CUDA0": 100},
            "compute_by_op_and_buffer": {"MUL_MAT": {"CUDA0": 100}},
        }
        CPE.validate_host_cert(cert, 12, 48)
        cert["compute_by_op_and_buffer"] = {"MUL_MAT": {"CPU": 1}}
        with self.assertRaisesRegex(CPE.LiveGateError, "fallback"):
            CPE.validate_host_cert(cert, 12, 48)

    def test_summary_accepts_matched_live_relief(self) -> None:
        rows = []
        for repeat in range(3):
            rows += [row("P0", repeat, 100.0, 10.0, False), row("P2", repeat, 80.0, 10.2, True)]
        result = CPE.summarize(rows, 1.05, 2.0, 0.1, 0.9)
        self.assertAlmostEqual(result["selected_gpu_energy_saving_frac"], 0.2)
        self.assertTrue(result["high_priority_gate"])
        self.assertTrue(result["server_board_relief_gate"])

    def test_summary_rejects_non_live_treatment(self) -> None:
        rows = []
        for repeat in range(3):
            rows += [row("P0", repeat, 100.0, 10.0, False), row("P2", repeat, 80.0, 10.0, False)]
        with self.assertRaisesRegex(CPE.LiveGateError, "live phone"):
            CPE.summarize(rows, 1.05, 2.0, 0.1, 0.9)

    def test_summary_rejects_unmatched_work(self) -> None:
        rows = []
        for repeat in range(3):
            rows += [row("P0", repeat, 100.0, 10.0, False), row("P2", repeat, 80.0, 10.0, True)]
        bad = copy.deepcopy(rows)
        bad[-1]["gemma"]["generated_tokens"] = 127
        with self.assertRaisesRegex(CPE.LiveGateError, "matched work"):
            CPE.summarize(bad, 1.05, 2.0, 0.1, 0.9)

    def test_high_priority_gate_is_load_bearing(self) -> None:
        rows = []
        for repeat in range(3):
            rows += [row("P0", repeat, 100.0, 10.0, False), row("P2", repeat, 80.0, 10.6, True)]
        self.assertFalse(CPE.summarize(rows, 1.05, 2.0, 0.1, 0.9)["high_priority_gate"])

    def test_overlap_gate_is_load_bearing(self) -> None:
        rows = []
        for repeat in range(3):
            rows += [row("P0", repeat, 100.0, 10.0, False), row("P2", repeat, 80.0, 10.0, True)]
        rows[-1]["concurrent_overlap_fraction_of_shorter"] = 0.89
        self.assertFalse(CPE.summarize(rows, 1.05, 2.0, 0.1, 0.9)["overlap_gate"])

    def test_low_priority_gate_is_load_bearing(self) -> None:
        rows = []
        for repeat in range(3):
            p0 = row("P0", repeat, 100.0, 10.0, False)
            p2 = row("P2", repeat, 80.0, 10.0, True)
            p2["gemma"]["lat_us_p95"] = 201.0
            rows += [p0, p2]
        self.assertFalse(CPE.summarize(rows, 1.05, 2.0, 0.1, 0.9)["low_priority_gate"])


if __name__ == "__main__":
    unittest.main()
