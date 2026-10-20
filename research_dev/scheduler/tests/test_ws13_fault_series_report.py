"""WS13: the fault-series report on a g11-shaped run (two recovered requests, mask-out, live re-attach)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler.campaigns.burstgpt.tools import fault_series_report as report  # noqa: E402

PAID_START = 1_000_000_000_000
LAUNCH = PAID_START - 3_000_000_000           # the co-helper started 3 s before the trace epoch


def us(seconds: float) -> int:
    return int(round(seconds * 1e6))


def g11_like(root: Path, *, fault: bool = True) -> Path:
    arm = root / "inputs-derived-mpaoa-p6f1flt"
    run = arm / "run-eval" / "run"
    run.mkdir(parents=True)
    requests = [{"request_id": f"r{index:03d}", "completion": {"actual_end_us": us(100 + index)}}
                for index in range(14)]
    recovered = [
        {"kind": "REQUEST_RECOVERED", "request": "burstgpt:003", "failed_at_us": us(505.6), "penalty_us": us(47.0),
         "tokens_discarded": 71, "quarantine_action": "mask_out", "attempt_started_us": us(458.6)},
        {"kind": "REQUEST_RECOVERED", "request": "burstgpt:004", "failed_at_us": us(505.6), "penalty_us": us(47.2),
         "tokens_discarded": 72, "quarantine_action": "mask_out", "attempt_started_us": us(458.4)},
    ]
    (run / "RESULT.json").write_text(json.dumps({
        "status": "PASS", "paid_start_ns": PAID_START, "paid_end_ns": PAID_START + 1_814_000_000_000,
        "request_results": requests, "request_recovered_events": recovered,
        "helper_mask_events": [
            {"kind": "HELPER_MASKED_OUT", "device": "pixel10pro-phone", "server": "qwen", "at_us": us(505.5)},
            {"kind": "HELPER_REATTACHED", "device": "pixel10pro-phone", "server": "qwen", "at_us": us(570.3)},
            {"kind": "HELPER_MASKED_OUT", "device": "op15-phone", "server": "gemma", "at_us": us(10.0)}],
        "device_membership_events": [
            {"kind": "DEVICE_QUARANTINED", "device_id": "pixel10pro-phone", "at_us": us(505.5)},
            {"kind": "DEVICE_READMITTED", "device_id": "pixel10pro-phone", "at_us": us(570.2)}],
    }))
    (run / "CO_HELPER_LIFECYCLE.json").write_text(json.dumps([
        {"phase": "launch", "device_id": "pixel10pro-phone", "monotonic_ns": LAUNCH, "worker_pids": [10263]},
        {"kind": "JOINED", "device_id": "pixel10pro-phone", "monotonic_ns": PAID_START + 570_100_000_000}]))
    if fault:
        (run / "FAULT_INJECTED.json").write_text(json.dumps({
            "status": "INJECTED", "when": {"kind": "active", "value": 500}, "signalled_pids": [10263],
            "condition": {"kind": "active", "seconds": 500, "trace_origin_monotonic_ns": LAUNCH,
                          "elapsed_s": 508.4},
            "timestamps": {"condition_met_wall_s": 1000.0, "signal_sent_wall_s": 1000.05}}))
    return arm


class FaultSeriesReportTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_g11_shape(self) -> None:
        row = report.fault_row("f1", g11_like(self.root), energy=False)
        self.assertAlmostEqual(row["injected_s"], 505.45, places=6)   # 508.4 - 3 s launch lead + 0.05 s
        self.assertAlmostEqual(row["detected_s"], 505.5)
        self.assertAlmostEqual(row["detection_delay_s"], 0.05, places=6)
        self.assertEqual(row["recovered_requests"], ["burstgpt:003", "burstgpt:004"])
        self.assertEqual((row["tokens_discarded"], row["servers_retired"]), (143, 0))
        self.assertAlmostEqual(row["failed_attempt_s"], 94.2)
        self.assertEqual(row["recovery_actions"], ["mask_out"])
        self.assertAlmostEqual(row["rejoin_after_loss_s"], 64.7, places=6)
        self.assertAlmostEqual(row["reattached_s"], 570.3)
        self.assertAlmostEqual(row["joined_s"], 570.1)
        self.assertEqual((row["completed"], row["requests"]), (14, 14))
        self.assertEqual(row["fault"]["status"], "INJECTED")

    def test_a_run_without_a_fault_record_is_reported_not_guessed(self) -> None:
        row = report.fault_row("f2", g11_like(self.root, fault=False), energy=False)
        self.assertIsNone(row["fault"])
        self.assertIsNone(row["injected_s"])
        self.assertIsNone(row["detection_delay_s"])
        self.assertIn("NO RECORD", report.markdown({"runs": [row]}))

    def test_energy_and_baseline_delta(self) -> None:
        arm = g11_like(self.root)
        base = self.root / "base"
        values = {"f1": 120_000.0, "u1": 100_000.0, "u2": 110_000.0}
        fake = lambda label, path, meter_path=None: {  # noqa: E731
            "fleet_j": values[label], "desktop": {"host_j": values[label] - 10_000.0}}
        with mock.patch.object(report.fleet_energy, "account_run", side_effect=fake):
            result = report.build_report({"f1": arm}, {"u1": base, "u2": base}, {})
        row = result["runs"][0]
        self.assertEqual((row["fleet_kj"], row["host_kj"]), (120.0, 110.0))
        self.assertAlmostEqual(row["fleet_delta_vs_baseline_kj"], 15.0)
        self.assertIn("| f1 | INJECTED active=500", report.markdown(result))


if __name__ == "__main__":
    unittest.main()
