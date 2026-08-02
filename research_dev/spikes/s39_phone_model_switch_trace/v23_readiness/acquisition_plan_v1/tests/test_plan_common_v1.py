#!/usr/bin/env python3

from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
PLAN = HERE.parent
sys.path.insert(0, str(PLAN))

import plan_common_v1 as common


class PlanCommonTests(unittest.TestCase):
    def test_historical_sources_remain_frozen_but_ineligible(self):
        common.verify_historical_readiness_sources()
        self.assertNotEqual(common.PLAN_SCHEMA, common.LEGACY_PLAN_SCHEMA)
        self.assertIn("production-v2", common.PLAN_SCHEMA)

    def test_manifest_round_trip(self):
        paths = [common.CANDIDATE, common.CONTRACT]
        raw = common.manifest_bytes(paths)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "SHA256SUMS.txt"
            manifest.write_bytes(raw)
            common.validate_manifest(manifest.resolve(), paths)

    def test_manifest_rejects_duplicate_escape_and_pycache(self):
        digest = "a" * 64
        cases = (
            f"{digest}  x\n{digest}  x\n",
            f"{digest}  ../x\n",
            f"{digest}  x/__pycache__/m.cpython-313.pyc\n",
        )
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(common.PlanError):
                    common.parse_manifest(raw.encode("ascii"))

    def test_non_stdlib_import_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "driver.py"
            path.write_text("import local_unbound_support\n", encoding="ascii")
            with self.assertRaisesRegex(common.PlanError, "E_UNBOUND_IMPORT"):
                common.verify_no_unbound_python_imports([path])

    def test_explicit_source_loaders_use_only_stdlib_imports(self):
        common.verify_no_unbound_python_imports(
            [
                common.HISTORICAL_ARTIFACT_DRIVER,
                common.HISTORICAL_FRESH_DRIVER,
                common.HISTORICAL_READINESS_SUPPORT,
            ]
        )

    def test_production_outputs_include_runtime_bundle_identity(self):
        self.assertEqual(len(common.PAYLOAD_ROLES), 10)
        self.assertEqual(
            common.OUTPUT_FILES["runtime_bundle_identity"],
            "runtime_bundle_identity.json",
        )


if __name__ == "__main__":
    unittest.main()
