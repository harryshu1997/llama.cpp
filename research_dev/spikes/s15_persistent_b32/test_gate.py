#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_gate  # noqa: E402
import validate_report  # noqa: E402


BOOT = "boot-id"


def cert() -> dict:
    return {
        "schema": "ls-stagenet-session-v2",
        "proto_version": 2,
        "session_id": 1,
        "session_end": "DETACH",
        "expected_backend": "HTP0",
        "worker_pid": 123,
        "worker_boot_nonce": "nonce",
        "device_boot_id": BOOT,
        "layer_start": 0,
        "layer_end": 8,
        "n_layer": 48,
        "steps_session": 40,
        "steps_total": 40,
        "reset_applied": True,
        "missing_buffer_compute_nodes": 0,
        "compute_by_op_and_buffer": {
            "MUL_MAT": {"HTP0": 10},
            "GET_ROWS": {"CPU": 1},
        },
        "placement_status": "SCHEDULED_PLACEMENT_OK",
    }


class SessionCertificateTests(unittest.TestCase):
    def test_valid_detach(self) -> None:
        self.assertEqual(run_gate.validate_session_cert(cert(), 1, "DETACH", BOOT, None), (123, "nonce"))

    def test_reset_false_is_rejected(self) -> None:
        value = cert()
        value["reset_applied"] = False
        with self.assertRaisesRegex(run_gate.GateError, "reset_applied"):
            run_gate.validate_session_cert(value, 1, "DETACH", BOOT, None)

    def test_changed_worker_identity_is_rejected(self) -> None:
        value = cert()
        with self.assertRaisesRegex(run_gate.GateError, "identity changed"):
            run_gate.validate_session_cert(value, 1, "DETACH", BOOT, (124, "nonce"))

    def test_cpu_fallback_is_rejected(self) -> None:
        value = cert()
        value["compute_by_op_and_buffer"]["MUL_MAT"] = {"CPU": 10}
        with self.assertRaisesRegex(run_gate.GateError, "fallback"):
            run_gate.validate_session_cert(value, 1, "DETACH", BOOT, None)

    def test_unknown_field_is_rejected(self) -> None:
        value = cert()
        value["extra"] = 1
        with self.assertRaisesRegex(run_gate.GateError, "key set"):
            run_gate.validate_session_cert(value, 1, "DETACH", BOOT, None)

    def test_stop_requires_no_reset(self) -> None:
        value = cert()
        value["session_end"] = "STOP"
        value["reset_applied"] = False
        run_gate.validate_session_cert(value, 1, "STOP", BOOT, None)

    def test_runner_refuses_without_run(self) -> None:
        process = subprocess.run(
            [sys.executable, str(HERE / "run_gate.py")], capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("physical execution requires --run", process.stderr)

    def test_persisted_result_validates_independently(self) -> None:
        report = validate_report.validate()
        self.assertEqual(report["verdict"], "PERSISTENT_OP15_B32_MECHANICS_PASS_ENERGY_UNKNOWN")

    def test_independent_validator_rejects_gapped_session(self) -> None:
        value = cert()
        value["session_id"] = 2
        with self.assertRaisesRegex(validate_report.ValidationError, "session_id"):
            validate_report.validate_certificate(value, 1, BOOT, 123, "nonce")


if __name__ == "__main__":
    unittest.main()
