#!/usr/bin/env python3

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
SCRIPT = S39 / "build_route_readiness.py"

SPEC = importlib.util.spec_from_file_location("s39_route_readiness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
READINESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READINESS)


class RouteReadinessTests(unittest.TestCase):
    def test_current_evidence_is_derived_honestly(self):
        result = READINESS.build(
            S39 / "results" / "w0_route_screen",
            S39 / "SHARD_MANIFEST.json",
        )
        routes = result["routes"]
        self.assertEqual(routes["gemma-4-12b-it-q4_0"]["status"], "FAIL_CORRECTNESS")
        self.assertEqual(routes["qwen3-14b-q4_k_m"]["status"], "PROVISIONAL_BATCH")
        self.assertNotIn("PASS", {route["status"] for route in routes.values()})

    def test_checked_in_readiness_matches_reducer(self):
        expected = READINESS.canonical_bytes(
            READINESS.build(
                S39 / "results" / "w0_route_screen",
                S39 / "SHARD_MANIFEST.json",
            )
        )
        self.assertEqual((S39 / "CURRENT_ROUTE_READINESS.json").read_bytes(), expected)

    def test_reducer_is_byte_deterministic(self):
        first = READINESS.canonical_bytes(
            READINESS.build(
                S39 / "results" / "w0_route_screen",
                S39 / "SHARD_MANIFEST.json",
            )
        )
        second = READINESS.canonical_bytes(
            READINESS.build(
                S39 / "results" / "w0_route_screen",
                S39 / "SHARD_MANIFEST.json",
            )
        )
        self.assertEqual(first, second)

    def test_corrupt_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_readiness_") as directory:
            root = Path(directory)
            shutil.copytree(S39 / "results" / "w0_route_screen", root / "evidence")
            path = root / "evidence" / "op15_qwen_headnet.log"
            path.write_bytes(path.read_bytes() + b"x")
            with self.assertRaisesRegex(READINESS.ReadinessError, "SHA-256 mismatch"):
                READINESS.build(root / "evidence", S39 / "SHARD_MANIFEST.json")

    def test_unlisted_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_readiness_") as directory:
            root = Path(directory)
            shutil.copytree(S39 / "results" / "w0_route_screen", root / "evidence")
            sums = root / "evidence" / "SHA256SUMS.txt"
            sums.write_text(
                sums.read_text(encoding="ascii")
                + "0" * 64
                + "  unbound.log\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(READINESS.ReadinessError, "artifact set mismatch"):
                READINESS.build(root / "evidence", S39 / "SHARD_MANIFEST.json")

    def test_float_layer_count_is_rejected(self):
        value = json.loads((S39 / "SHARD_MANIFEST.json").read_text(encoding="ascii"))
        value["models"]["qwen3-14b-q4_k_m"]["source"]["block_count"] = 40.0
        with tempfile.TemporaryDirectory(prefix="s39_readiness_") as directory:
            path = Path(directory) / "shards.json"
            path.write_bytes(READINESS.canonical_bytes(value))
            with self.assertRaisesRegex(
                READINESS.ReadinessError,
                "layers|block_count",
            ):
                READINESS.build(S39 / "results" / "w0_route_screen", path)

    def test_stale_batch_certificate_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_readiness_") as directory:
            root = Path(directory)
            shutil.copytree(
                S39 / "results" / "w0_qwen_batch_wifi",
                root / "batch",
            )
            path = root / "batch" / "qwen_batch_certificate.json"
            value = json.loads(path.read_text(encoding="ascii"))
            value["cohorts"][2]["cohort_requests"] = 31
            path.write_bytes(READINESS.canonical_bytes(value))
            with self.assertRaisesRegex(
                READINESS.ReadinessError,
                "does not match raw evidence",
            ):
                READINESS.build(
                    S39 / "results" / "w0_route_screen",
                    S39 / "SHARD_MANIFEST.json",
                    root / "batch",
                )


if __name__ == "__main__":
    unittest.main()
