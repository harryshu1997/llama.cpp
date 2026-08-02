#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
S41 = HERE.parent
S40_EXECUTORS = (
    S41.parent / "s40_shared_warm_tier_server" / "executors"
)
for path in (S40_EXECUTORS, S41):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_desktop_smoke_config as bundle  # noqa: E402
from phone_gateway import GatewayError, strict_json_loads  # noqa: E402
import desktop_authority as authority  # noqa: E402


class S41BindingTests(unittest.TestCase):
    def contract(self):
        raw = authority.CONTRACT_BYTES
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            authority.CONTRACT_SHA256,
        )
        value = strict_json_loads(raw, "S41 test contract")
        return raw, value

    def test_contract_binds_exact_pair_runtime_and_serving(self):
        _, value = self.contract()
        result = authority.validate_contract_value(value)
        self.assertEqual(result["pair"], list(authority.EXPECTED_PAIR))
        self.assertEqual(result["models"], authority.EXPECTED_MODELS)
        self.assertEqual(result["runtime"], authority.EXPECTED_RUNTIME)
        self.assertEqual(result["serving"], authority.EXPECTED_SERVING)

    def test_contract_rejects_pair_or_runtime_substitution(self):
        _, value = self.contract()
        changed_pair = copy.deepcopy(value)
        changed_pair["pair"][0] = "qwen3-8b-q8_0"
        with self.assertRaisesRegex(GatewayError, "S41 model pair"):
            authority.validate_contract_value(changed_pair)
        changed_runtime = copy.deepcopy(value)
        changed_runtime["runtime"]["llama_server"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(GatewayError, "S41 runtime binding"):
            authority.validate_contract_value(changed_runtime)

    def test_bundle_overrides_only_contract_and_authority(self):
        changed = {
            name
            for name in bundle.SOURCES
            if bundle.SOURCES[name] != bundle.S40_SOURCES[name]
        }
        self.assertEqual(
            changed,
            {"qualification_authority.py"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "executor-bundle"
            result = bundle.build_executor_bundle(output)
            self.assertEqual(result["status"], "PASS")
            manifest = json.loads((output / "MANIFEST.json").read_bytes())
            rows = {row["name"]: row for row in manifest["files"]}
            self.assertEqual(
                rows["experiment/EXPERIMENT_CONTRACT.json"]["source_path"],
                str(S41 / "desktop_authority.py"),
            )
            self.assertEqual(
                rows["qualification_authority.py"]["source_path"],
                str(S41 / "desktop_authority.py"),
            )


if __name__ == "__main__":
    unittest.main()
