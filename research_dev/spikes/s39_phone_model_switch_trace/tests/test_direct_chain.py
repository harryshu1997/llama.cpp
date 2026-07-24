#!/usr/bin/env python3

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
SCRIPT = S39 / "summarize_direct_chain.py"
EVIDENCE = S39 / "results" / "w1_direct_phone_chain"

SPEC = importlib.util.spec_from_file_location("s39_direct_chain", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class DirectChainTests(unittest.TestCase):
    def copy_evidence(self, root: Path) -> Path:
        destination = root / "evidence"
        shutil.copytree(EVIDENCE, destination)
        return destination

    def rewrite_json(self, path: Path, mutate) -> None:
        value = json.loads(path.read_text(encoding="ascii"))
        mutate(value)
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )

    def test_current_evidence_passes_mechanics_only(self):
        result = SUMMARY.build()
        self.assertEqual(
            result["status"],
            "DIRECT_CHAIN_MECHANICS_PASS_REPEATS_PENDING",
        )
        self.assertEqual(result["correctness"]["token_checks"], 264)
        self.assertEqual(
            result["direct_cohorts"][1]["direct_activation_payload_bytes"],
            7_864_320,
        )
        self.assertEqual(
            result["transport"]["reservation_binding"],
            "IMPLICIT_SINGLE_CLIENT_MECHANICS_ONLY",
        )

    def test_checked_in_certificate_matches_reducer(self):
        self.assertEqual(
            (EVIDENCE / "direct_chain_certificate.json").read_bytes(),
            SUMMARY.common.canonical_bytes(SUMMARY.build()),
        )

    def test_token_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_direct_") as directory:
            evidence = self.copy_evidence(Path(directory))
            self.rewrite_json(
                evidence / "b32_fixed.json",
                lambda value: value["requests"][4]["tokens"].__setitem__(0, 1),
            )
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "tokens",
            ):
                SUMMARY.build(evidence)

    def test_host_payload_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_direct_") as directory:
            evidence = self.copy_evidence(Path(directory))
            self.rewrite_json(
                evidence / "b1_fixed.json",
                lambda value: value["transport"].__setitem__(
                    "host_activation_payload_bytes",
                    False,
                ),
            )
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "transport",
            ):
                SUMMARY.build(evidence)

    def test_relay_endpoint_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_direct_") as directory:
            evidence = self.copy_evidence(Path(directory))
            path = evidence / "op15_relay_fixed_b32.log"
            raw = path.read_bytes().replace(
                b'"tail_endpoint":"172.20.59.72:39312"',
                b'"tail_endpoint":"127.0.0.1:39312"',
            )
            path.write_bytes(raw)
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "tail_endpoint",
            ):
                SUMMARY.build(evidence)

    def test_worker_identity_change_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_direct_") as directory:
            evidence = self.copy_evidence(Path(directory))
            path = evidence / "op15_fixed_head.log"
            raw = path.read_bytes().replace(
                b'"worker_pid":11104',
                b'"worker_pid":11105',
                1,
            )
            path.write_bytes(raw)
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "worker_pid",
            ):
                SUMMARY.build(evidence)

    def test_host_control_token_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_direct_") as directory:
            evidence = self.copy_evidence(Path(directory))
            self.rewrite_json(
                evidence / "matched_host_relay_b1.json",
                lambda value: value["requests"][0]["tokens"].__setitem__(0, 1),
            )
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "tokens",
            ):
                SUMMARY.build(evidence)

    def test_cli_fails_closed_without_traceback(self):
        with tempfile.TemporaryDirectory(prefix="s39_direct_") as directory:
            process = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--evidence",
                    directory,
                    "--output",
                    str(Path(directory) / "out.json"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(process.returncode, 2)
        self.assertIn("S39_DIRECT_EVIDENCE_ERROR", process.stderr)
        self.assertNotIn("Traceback", process.stderr)
        self.assertEqual(process.stdout, "")


if __name__ == "__main__":
    unittest.main()
