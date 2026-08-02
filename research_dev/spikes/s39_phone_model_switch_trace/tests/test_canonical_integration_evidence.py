#!/usr/bin/env python3

import importlib.util
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
SCRIPT = S39 / "summarize_direct_mixed.py"
EVIDENCE = S39 / "results" / "w3_canonical_integration"

SPEC = importlib.util.spec_from_file_location("s39_canonical_integration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class CanonicalIntegrationEvidenceTests(unittest.TestCase):
    def test_real_mixed_route_passes_with_integrated_batcher(self):
        result = SUMMARY.build(EVIDENCE)
        self.assertEqual(
            result["status"],
            "DIRECT_MIXED_BATCH_MECHANICS_PASS",
        )
        self.assertEqual(result["correctness"]["token_checks"], 256)
        self.assertEqual(
            result["execution"]["batch_sizes"],
            SUMMARY.BATCH_SIZES,
        )

    def test_checked_in_certificate_matches_reducer(self):
        self.assertEqual(
            (EVIDENCE / "direct_mixed_certificate.json").read_bytes(),
            SUMMARY.common.canonical_bytes(SUMMARY.build(EVIDENCE)),
        )

    def test_source_binding_matches_executed_files(self):
        context = json.loads(
            (EVIDENCE / "RUN_CONTEXT.json").read_text(encoding="ascii")
        )
        source = context["source"]
        self.assertEqual(
            source["base_git_commit"],
            "7d1926dffc0e9666ec5aa507e688727048827676",
        )
        self.assertEqual(
            source["mixed_phase_batcher_sha256"],
            SUMMARY.common.sha256((S39 / "mixed_phase_batcher.py").read_bytes()),
        )
        self.assertEqual(
            source["direct_mixed_probe_sha256"],
            SUMMARY.common.sha256((S39 / "direct_mixed_probe.py").read_bytes()),
        )
        self.assertEqual(
            source["mixed_phase_batcher_patch_sha256"],
            SUMMARY.common.sha256(
                (EVIDENCE / "mixed_phase_batcher.patch").read_bytes()
            ),
        )


if __name__ == "__main__":
    unittest.main()
