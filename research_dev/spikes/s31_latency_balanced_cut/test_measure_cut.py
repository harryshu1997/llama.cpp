#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from measure_cut import validate_launch_evidence
from physical_adapter import PhysicalRuntimeError


MODEL_SHA256 = "7b56cbd0e0d96d5c8d7df9c21b39d67264e58cbabf7a96d4681eff8b3d492848"


def write_session_files(root: Path, desktop_sha256: str) -> tuple[Path, Path]:
    phone = root / "phone.env"
    phone.write_text(
        "schema=s24-a6000-phone-session-v1\n"
        "runtime_activation_relay=DESKTOP_DIRECT_WIFI\n"
        "max_streams=32\n"
        "context=16\n"
        "op12_layer_range=0:3\n"
        "op15_layer_range=3:8\n"
        f"op12_head_sha256={MODEL_SHA256}\n"
        f"op15_mid_sha256={MODEL_SHA256}\n",
        encoding="ascii",
    )
    desktop = root / "desktop.env"
    desktop.write_text(
        "schema=s24-desktop-cuda-session-v1\n"
        "context=16\n"
        "prefix_streams=32\n"
        "mid_streams=32\n"
        "tail_streams=32\n"
        "prefix_layer_range=0:6\n"
        "mid_layer_range=6:8\n"
        "tail_layer_range=8:48\n"
        f"model_sha256={desktop_sha256}\n",
        encoding="ascii",
    )
    return phone, desktop


class LaunchEvidenceTests(unittest.TestCase):
    def test_accepts_one_shared_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            phone, desktop = write_session_files(Path(directory), MODEL_SHA256)
            result = validate_launch_evidence(phone, desktop, 3)
            self.assertEqual(result["desktop_model_sha256"], MODEL_SHA256)
            self.assertEqual(result["op12_model_sha256"], MODEL_SHA256)
            self.assertEqual(result["op15_model_sha256"], MODEL_SHA256)

    def test_rejects_desktop_model_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            phone, desktop = write_session_files(Path(directory), "a" * 64)
            with self.assertRaisesRegex(PhysicalRuntimeError, "identities differ"):
                validate_launch_evidence(phone, desktop, 3)


if __name__ == "__main__":
    unittest.main()
