#!/usr/bin/env python3
"""Offline tests for the stock-default control contract."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


import validate_contract
import run_dual_default
import run_default_campaign


class DefaultControlTests(unittest.TestCase):
    def test_contract_validates(self) -> None:
        result = validate_contract.validate()
        self.assertEqual(result["status"], "STOCK_DEFAULT_CONTRACT_VALID")
        self.assertEqual(result["request_count"], 74)

    def test_manifest_excludes_itself_and_is_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "z.json").write_text("{}\n", encoding="ascii")
            (path / "a.json").write_text("{}\n", encoding="ascii")
            run_dual_default.write_manifest(path)
            lines = (path / "SHA256SUMS.txt").read_text(
                encoding="ascii"
            ).splitlines()
            self.assertEqual(
                [line.split("  ", 1)[1] for line in lines],
                ["a.json", "z.json"],
            )

    def test_campaign_manifest_is_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "nested").mkdir()
            (path / "nested" / "value.json").write_text(
                "{}\n", encoding="ascii"
            )
            run_default_campaign.write_manifest(path)
            line = (path / "SHA256SUMS.txt").read_text(encoding="ascii")
            self.assertIn("  nested/value.json\n", line)

    def test_contract_forbids_all_explicit_tuning_options(self) -> None:
        contract = validate_contract.load_contract()
        forbidden = set(contract["tuning_arguments_forbidden"])
        self.assertIn("--parallel", forbidden)
        self.assertIn("--n-gpu-layers", forbidden)
        self.assertIn("--flash-attn", forbidden)
        self.assertNotIn("--model", forbidden)

    def test_contract_is_canonical_json(self) -> None:
        path = validate_contract.HERE / "DEFAULT_CONTROL_CONTRACT.json"
        value = json.loads(path.read_text(encoding="ascii"))
        expected = (
            json.dumps(
                value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True,
            ) + "\n"
        ).encode("ascii")
        self.assertEqual(path.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
