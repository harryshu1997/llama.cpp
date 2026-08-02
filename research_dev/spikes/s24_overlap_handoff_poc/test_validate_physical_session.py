#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from validate_physical_session import (
    ValidationError,
    parse_session_certificates,
    validate_backend_tally,
    validate_certificate,
)


class PlacementValidationTests(unittest.TestCase):
    def test_session_parser_accepts_prefixed_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.log"
            path.write_text(
                "prefix SESSIONCERT {\"session_id\":1}\n",
                encoding="ascii",
            )
            self.assertEqual(parse_session_certificates(path), [{"session_id": 1}])

    def test_declared_get_rows_seams_are_narrow(self) -> None:
        exceptions = validate_backend_tally(
            "op12-prefix",
            {"MUL_MAT": {"HTP0": 10}, "GET_ROWS": {"CPU": 2}},
            "HTP0",
        )
        self.assertEqual(len(exceptions), 1)
        with self.assertRaisesRegex(ValidationError, "undeclared"):
            validate_backend_tally(
                "op12-prefix", {"MUL_MAT": {"CPU": 10}}, "HTP0",
            )

    def test_idle_worker_requires_unobserved_certificate(self) -> None:
        certificate = {
            "schema": "ls-stagenet-session-v2",
            "proto_version": 2,
            "session_id": 1,
            "session_end": "DETACH",
            "expected_backend": "HTP0",
            "worker_pid": 10,
            "worker_boot_nonce": "abc",
            "device_boot_id": "boot",
            "layer_start": 0,
            "layer_end": 8,
            "n_layer": 48,
            "steps_session": 0,
            "steps_total": 0,
            "reset_applied": True,
            "missing_buffer_compute_nodes": 0,
            "compute_by_op_and_buffer": {},
            "placement_status": "PLACEMENT_UNOBSERVED",
        }
        result = validate_certificate("op12-prefix", certificate, 0, "detach")
        self.assertEqual(result["placement_status"], "PLACEMENT_UNOBSERVED")


if __name__ == "__main__":
    unittest.main()
