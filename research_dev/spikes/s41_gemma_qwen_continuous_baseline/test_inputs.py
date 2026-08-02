#!/usr/bin/env python3
"""Focused tests for the versioned S41 workload inputs."""

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

    def copy_tree(self) -> Path:
        temporary = Path(tempfile.mkdtemp(prefix="s41-input-test-"))
        self.addCleanup(shutil.rmtree, temporary)
        spike_root = temporary / "spikes"
        root = spike_root / HERE.name
        root.mkdir(parents=True)
        for name in (
            "build_inputs.py",
            "REQUESTS.jsonl",
            "SWITCHES.jsonl",
            "INPUT_CONTRACT.json",
            "INPUT_MANIFEST.json",
        ):
            shutil.copy2(HERE / name, root / name)
        for source_dir, names in (
            (
                "s39_desktop_swap_baseline",
                (
                    "DESKTOP_BASELINE_CONTRACT.json",
                    "DESKTOP_REQUESTS.jsonl",
                    "DESKTOP_SWITCHES.jsonl",
                    "INPUT_MANIFEST.json",
                ),
            ),
            (
                "s39_phone_model_switch_trace",
                ("ACTIVE_TRACE.json",),
            ),
        ):
            target = spike_root / source_dir
            target.mkdir()
            real = HERE.parent / source_dir
            for name in names:
                shutil.copy2(real / name, target / name)
        bundle = spike_root / "s39_phone_model_switch_trace/bundle_frequent"
        bundle.mkdir()
        real_bundle = (
            HERE.parent / "s39_phone_model_switch_trace/bundle_frequent")
        for name in ("requests.jsonl", "replay_intents.jsonl"):
            shutil.copy2(real_bundle / name, bundle / name)
        return root

    @staticmethod
    def rebind(root: Path, names: tuple[str, ...]) -> None:
        manifest_path = root / "INPUT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        for name in names:
            data = (root / name).read_bytes()
            manifest["files"][name]["bytes"] = len(data)
            manifest["files"][name]["sha256"] = validate.sha256(data)
        manifest_path.write_bytes(validate.canonical(manifest))

    def test_valid_bundle(self) -> None:
        report = validate.validate(HERE)
        self.assertEqual(report["status"], "S41_INPUTS_VALID")
        self.assertEqual(report["request_count"], 74)
        self.assertEqual(report["switch_count"], 9)
        self.assertEqual(report["maximum_token_id"], 84831)

    def test_builder_is_byte_deterministic(self) -> None:
        paths = [
            HERE / name for name in (
                "REQUESTS.jsonl",
                "SWITCHES.jsonl",
                "INPUT_CONTRACT.json",
                "INPUT_MANIFEST.json",
            )
        ]
        before = {path.name: path.read_bytes() for path in paths}
        subprocess.run(
            [sys.executable, str(HERE / "build_inputs.py")],
            cwd=HERE,
            check=True,
            capture_output=True,
        )
        self.assertEqual(
            before, {path.name: path.read_bytes() for path in paths})

    def test_geometry_and_token_arrays_match_s39(self) -> None:
        source = [
            json.loads(line) for line in (
                HERE.parent
                / "s39_desktop_swap_baseline/DESKTOP_REQUESTS.jsonl"
            ).read_text(encoding="ascii").splitlines()
        ]
        output = [
            json.loads(line) for line in (
                HERE / "REQUESTS.jsonl"
            ).read_text(encoding="ascii").splitlines()
        ]
        for old, new in zip(source, output):
            self.assertEqual(old["event_id"], new["event_id"])
            self.assertEqual(old["arrival_us"], new["arrival_us"])
            self.assertEqual(old["input_tokens"], new["input_tokens"])
            self.assertEqual(old["prompt_tokens"], new["prompt_tokens"])
            expected = validate.MODEL_REMAP[old["model_id"]]
            self.assertEqual(new["model_id"], expected)

    def test_every_token_is_valid_for_both_vocabularies(self) -> None:
        rows = [
            json.loads(line) for line in (
                HERE / "REQUESTS.jsonl"
            ).read_text(encoding="ascii").splitlines()
        ]
        limit = min(
            item["vocabulary_size"]
            for item in validate.EXPECTED_MODELS.values())
        self.assertTrue(all(
            0 <= token < limit
            for row in rows
            for token in row["prompt_tokens"]
        ))

    def test_manifest_tamper_is_rejected(self) -> None:
        root = self.copy_tree()
        path = root / "REQUESTS.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(validate.ValidationError):
            validate.validate(root)

    def test_geometry_tamper_is_rejected_after_rehash(self) -> None:
        root = self.copy_tree()
        request_path = root / "REQUESTS.jsonl"
        rows = [
            json.loads(line)
            for line in request_path.read_text(encoding="ascii").splitlines()
        ]
        rows[0]["arrival_us"] += 1
        request_data = b"".join(validate.canonical(row) for row in rows)
        request_path.write_bytes(request_data)
        contract_path = root / "INPUT_CONTRACT.json"
        contract = json.loads(contract_path.read_text(encoding="ascii"))
        contract["workload"]["requests_sha256"] = validate.sha256(request_data)
        contract_path.write_bytes(validate.canonical(contract))
        self.rebind(root, ("REQUESTS.jsonl", "INPUT_CONTRACT.json"))
        with self.assertRaisesRegex(
                validate.ValidationError, "source geometry changed"):
            validate.validate(root)

    def test_model_binding_tamper_is_rejected_after_rehash(self) -> None:
        root = self.copy_tree()
        contract_path = root / "INPUT_CONTRACT.json"
        contract = json.loads(contract_path.read_text(encoding="ascii"))
        contract["models"][validate.GEMMA]["bytes"] -= 1
        contract_path.write_bytes(validate.canonical(contract))
        self.rebind(root, ("INPUT_CONTRACT.json",))
        with self.assertRaisesRegex(
                validate.ValidationError, "model artifact binding changed"):
            validate.validate(root)

    def test_source_binding_tamper_is_rejected_after_rehash(self) -> None:
        root = self.copy_tree()
        contract_path = root / "INPUT_CONTRACT.json"
        contract = json.loads(contract_path.read_text(encoding="ascii"))
        key = "../s39_desktop_swap_baseline/DESKTOP_REQUESTS.jsonl"
        contract["source"]["artifacts"][key]["sha256"] = "0" * 64
        contract_path.write_bytes(validate.canonical(contract))
        self.rebind(root, ("INPUT_CONTRACT.json",))
        with self.assertRaisesRegex(
                validate.ValidationError, "source artifact binding changed"):
            validate.validate(root)


if __name__ == "__main__":
    unittest.main()
