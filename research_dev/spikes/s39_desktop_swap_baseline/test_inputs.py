#!/usr/bin/env python3
"""Tests for the CP0-D frozen input bundle."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import validate_inputs as validate  # noqa: E402


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            [sys.executable, str(HERE / "build_inputs.py")],
            cwd=HERE,
            check=True,
            capture_output=True,
        )

    def copy_bundle(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="cp0d-input-test-"))
        self.addCleanup(shutil.rmtree, root)
        for name in (
            "DESKTOP_REQUESTS.jsonl",
            "DESKTOP_SWITCHES.jsonl",
            "DESKTOP_BASELINE_CONTRACT.json",
            "INPUT_MANIFEST.json",
        ):
            shutil.copy2(HERE / name, root / name)
        return root

    def test_valid_bundle(self) -> None:
        report = validate.validate(HERE)
        self.assertEqual(report["status"], "CP0D_INPUTS_VALID")
        self.assertEqual(report["request_count"], 74)
        self.assertEqual(report["switch_count"], 9)

    def test_builder_is_byte_deterministic(self) -> None:
        before = {
            path.name: path.read_bytes()
            for path in (
                HERE / "DESKTOP_REQUESTS.jsonl",
                HERE / "DESKTOP_SWITCHES.jsonl",
                HERE / "DESKTOP_BASELINE_CONTRACT.json",
                HERE / "INPUT_MANIFEST.json",
            )
        }
        subprocess.run(
            [sys.executable, str(HERE / "build_inputs.py")],
            cwd=HERE,
            check=True,
            capture_output=True,
        )
        self.assertEqual(before, {name: (HERE / name).read_bytes() for name in before})

    def test_manifest_tamper_rejected(self) -> None:
        root = self.copy_bundle()
        path = root / "DESKTOP_REQUESTS.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(validate.ValidationError):
            validate.validate(root)

    def test_payload_length_tamper_rejected_after_rehash(self) -> None:
        root = self.copy_bundle()
        path = root / "DESKTOP_REQUESTS.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["prompt_tokens"].pop()
        data = b"".join(validate.canonical(row) for row in rows)
        path.write_bytes(data)
        contract_path = root / "DESKTOP_BASELINE_CONTRACT.json"
        contract = json.loads(contract_path.read_text())
        contract["replay"]["request_sha256"] = validate.sha(data)
        contract_data = validate.canonical(contract)
        contract_path.write_bytes(contract_data)
        manifest_path = root / "INPUT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][path.name]["bytes"] = len(data)
        manifest["files"][path.name]["sha256"] = validate.sha(data)
        manifest["files"][contract_path.name]["bytes"] = len(contract_data)
        manifest["files"][contract_path.name]["sha256"] = validate.sha(contract_data)
        manifest_path.write_bytes(validate.canonical(manifest))
        with self.assertRaises(validate.ValidationError):
            validate.validate(root)

    def test_switch_reorder_rejected_after_rehash(self) -> None:
        root = self.copy_bundle()
        path = root / "DESKTOP_SWITCHES.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0], rows[1] = rows[1], rows[0]
        data = b"".join(validate.canonical(row) for row in rows)
        path.write_bytes(data)
        contract_path = root / "DESKTOP_BASELINE_CONTRACT.json"
        contract = json.loads(contract_path.read_text())
        contract["replay"]["switch_sha256"] = validate.sha(data)
        contract_data = validate.canonical(contract)
        contract_path.write_bytes(contract_data)
        manifest_path = root / "INPUT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][path.name]["bytes"] = len(data)
        manifest["files"][path.name]["sha256"] = validate.sha(data)
        manifest["files"][contract_path.name]["bytes"] = len(contract_data)
        manifest["files"][contract_path.name]["sha256"] = validate.sha(contract_data)
        manifest_path.write_bytes(validate.canonical(manifest))
        with self.assertRaises(validate.ValidationError):
            validate.validate(root)


if __name__ == "__main__":
    unittest.main()
