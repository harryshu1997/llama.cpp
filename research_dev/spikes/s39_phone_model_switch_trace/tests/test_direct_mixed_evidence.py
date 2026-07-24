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
SCRIPT = S39 / "summarize_direct_mixed.py"
EVIDENCE = S39 / "results" / "w2_direct_mixed"

SPEC = importlib.util.spec_from_file_location("s39_direct_mixed", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class DirectMixedEvidenceTests(unittest.TestCase):
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

    def test_current_evidence_passes_mechanics(self):
        result = SUMMARY.build()
        self.assertEqual(
            result["status"],
            "DIRECT_MIXED_BATCH_MECHANICS_PASS",
        )
        self.assertEqual(result["correctness"]["token_checks"], 256)
        self.assertEqual(result["execution"]["batch_sizes"], SUMMARY.BATCH_SIZES)
        self.assertEqual(result["execution"]["mixed_decode_rows"], 16)
        self.assertEqual(result["execution"]["mixed_prefill_rows"], 80)

    def test_checked_in_certificate_matches_reducer(self):
        self.assertEqual(
            (EVIDENCE / "direct_mixed_certificate.json").read_bytes(),
            SUMMARY.common.canonical_bytes(SUMMARY.build()),
        )

    def test_token_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_mixed_") as directory:
            evidence = self.copy_evidence(Path(directory))
            self.rewrite_json(
                evidence / "mixed16x16.json",
                lambda value: value["requests"][0]["tokens"].__setitem__(0, 1),
            )
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "tokens",
            ):
                SUMMARY.build(evidence)

    def test_mixed_phase_order_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_mixed_") as directory:
            evidence = self.copy_evidence(Path(directory))
            self.rewrite_json(
                evidence / "mixed16x16.json",
                lambda value: value["batch_events"][1]["phases"].__setitem__(
                    0,
                    "prefill",
                ),
            )
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "phases",
            ):
                SUMMARY.build(evidence)

    def test_host_payload_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_mixed_") as directory:
            evidence = self.copy_evidence(Path(directory))
            self.rewrite_json(
                evidence / "mixed16x16.json",
                lambda value: value["transport"].__setitem__(
                    "host_activation_payload_bytes",
                    1,
                ),
            )
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "transport",
            ):
                SUMMARY.build(evidence)

    def test_relay_endpoint_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_mixed_") as directory:
            evidence = self.copy_evidence(Path(directory))
            path = evidence / "op15_relay.log"
            path.write_bytes(
                path.read_bytes().replace(
                    b'"tail_endpoint":"172.20.59.72:39312"',
                    b'"tail_endpoint":"127.0.0.1:39312"',
                )
            )
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "tail_endpoint",
            ):
                SUMMARY.build(evidence)

    def test_worker_step_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_mixed_") as directory:
            evidence = self.copy_evidence(Path(directory))
            path = evidence / "op12_tail.log"
            path.write_bytes(
                path.read_bytes().replace(
                    b'"steps_session":384',
                    b'"steps_session":385',
                    1,
                )
            )
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "steps_session",
            ):
                SUMMARY.build(evidence)

    def test_cpu_fallback_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_mixed_") as directory:
            evidence = self.copy_evidence(Path(directory))
            path = evidence / "op12_tail.log"
            path.write_bytes(
                path.read_bytes().replace(
                    b'"ADD":{"OpenCL":200}',
                    b'"ADD":{"CPU":200}',
                    1,
                )
            )
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "CPU fallback",
            ):
                SUMMARY.build(evidence)

    def test_bad_copy_count_is_rejected_without_traceback(self):
        with tempfile.TemporaryDirectory(prefix="s39_mixed_") as directory:
            evidence = self.copy_evidence(Path(directory))
            path = evidence / "op15_head.log"
            path.write_bytes(
                path.read_bytes().replace(
                    b'"copy_by_buffer_type":{}',
                    b'"copy_by_buffer_type":{"OpenCL":"bad"}',
                    1,
                )
            )
            process = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--evidence",
                    str(evidence),
                    "--output",
                    str(evidence / "output.json"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(process.returncode, 2)
        self.assertIn("S39_DIRECT_MIXED_EVIDENCE_ERROR", process.stderr)
        self.assertNotIn("Traceback", process.stderr)

    def test_cli_fails_closed_without_traceback(self):
        with tempfile.TemporaryDirectory(prefix="s39_mixed_") as directory:
            process = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--evidence",
                    directory,
                    "--output",
                    str(Path(directory) / "output.json"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(process.returncode, 2)
        self.assertIn("S39_DIRECT_MIXED_EVIDENCE_ERROR", process.stderr)
        self.assertNotIn("Traceback", process.stderr)
        self.assertEqual(process.stdout, "")


if __name__ == "__main__":
    unittest.main()
