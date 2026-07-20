#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import validate_report as validator


class ReportValidationTests(unittest.TestCase):
    def test_frozen_report_passes(self) -> None:
        value = validator.validate()
        self.assertEqual(value["problems"], [])

    def mutate_report(self, change) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            shutil.copytree(validator.RESULTS, root)
            report_path = root / "report.json"
            value = json.loads(report_path.read_bytes())
            change(value)
            report_path.write_bytes(validator.canonical(value))
            with self.assertRaises(validator.ValidationError):
                validator.validate(report_path, root)

    def test_phone_worker_mutation_is_rejected(self) -> None:
        self.mutate_report(
            lambda value: value["phone_sessions"][1].__setitem__("worker_pid", 9),
        )

    def test_host_backend_mutation_is_rejected(self) -> None:
        def change(value) -> None:
            value["host_placements"][0]["compute_by_buffer_type"] = {"CPU": 9288}

        self.mutate_report(change)

    def test_token_and_energy_claim_mutations_are_rejected(self) -> None:
        def tokens(value) -> None:
            value["child_results"][0]["token_ids"][0][0] += 1

        self.mutate_report(tokens)
        self.mutate_report(
            lambda value: value.__setitem__("formal_energy_claim", "TOTAL_SYSTEM_PASS"),
        )

    def test_bound_child_artifact_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            shutil.copytree(validator.RESULTS, root)
            path = root / "bridge-child-artifacts/launch-000001/tokens.bin"
            value = json.loads(path.read_bytes())
            value["token_ids"][0][0] += 1
            path.write_bytes(validator.canonical(value))
            with self.assertRaisesRegex(validator.ValidationError, "artifact binding"):
                validator.validate(root / "report.json", root)


if __name__ == "__main__":
    unittest.main()
