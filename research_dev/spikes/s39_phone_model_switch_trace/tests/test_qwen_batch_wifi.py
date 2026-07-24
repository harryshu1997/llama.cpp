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
SCRIPT = S39 / "summarize_qwen_batch_wifi.py"
EVIDENCE = S39 / "results" / "w0_qwen_batch_wifi"

SPEC = importlib.util.spec_from_file_location("s39_qwen_batch_wifi", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class QwenBatchWifiTests(unittest.TestCase):
    def test_current_evidence_is_provisional_and_exact(self):
        result = SUMMARY.build()
        self.assertEqual(result["status"], "PROVISIONAL_BATCH")
        self.assertEqual([row["cohort_requests"] for row in result["cohorts"]], [1, 8, 32])
        self.assertEqual(result["correctness"]["generated_token_checks"], 328)
        self.assertEqual(
            result["cohorts"][2]["activation_payload_bytes_total"],
            15_728_640,
        )
        self.assertEqual(
            result["transport"]["path_provenance"],
            "posthoc_operator_record_no_interface_counter",
        )

    def test_checked_in_certificate_matches_reducer(self):
        expected = SUMMARY.canonical_bytes(SUMMARY.build())
        self.assertEqual(
            (EVIDENCE / "qwen_batch_certificate.json").read_bytes(),
            expected,
        )

    def test_reducer_is_byte_deterministic(self):
        self.assertEqual(
            SUMMARY.canonical_bytes(SUMMARY.build()),
            SUMMARY.canonical_bytes(SUMMARY.build()),
        )

    def test_token_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_batch_") as directory:
            root = Path(directory)
            shutil.copytree(EVIDENCE, root / "evidence")
            path = root / "evidence" / "b8.json"
            value = json.loads(path.read_text(encoding="ascii"))
            value["requests"][3]["tokens"][0] += 1
            path.write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(SUMMARY.BatchEvidenceError, "tokens"):
                SUMMARY.build(root / "evidence")

    def test_batch_shape_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_batch_") as directory:
            root = Path(directory)
            shutil.copytree(EVIDENCE, root / "evidence")
            path = root / "evidence" / "b32.json"
            value = json.loads(path.read_text(encoding="ascii"))
            value["batch_events"]["head"][0]["batch_size"] = 159
            path.write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(SUMMARY.BatchEvidenceError, "batch_size"):
                SUMMARY.build(root / "evidence")

    def test_worker_epoch_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_batch_") as directory:
            root = Path(directory)
            shutil.copytree(EVIDENCE, root / "evidence")
            path = root / "evidence" / "op12_qwen_batch32_tail.log"
            raw = path.read_bytes()
            raw = raw.replace(b'"session_id":2', b'"session_id":9', 1)
            path.write_bytes(raw)
            with self.assertRaisesRegex(SUMMARY.BatchEvidenceError, "session_id"):
                SUMMARY.build(root / "evidence")

    def test_transport_claim_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_batch_") as directory:
            root = Path(directory)
            shutil.copytree(EVIDENCE, root / "evidence")
            path = root / "evidence" / "RUN_CONTEXT.json"
            value = json.loads(path.read_text(encoding="ascii"))
            value["runtime"]["direct_phone_to_phone"] = True
            path.write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                SUMMARY.BatchEvidenceError,
                "direct_phone_to_phone",
            ):
                SUMMARY.build(root / "evidence")

    def test_cli_fails_without_traceback_on_missing_input(self):
        with tempfile.TemporaryDirectory(prefix="s39_batch_") as directory:
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
        self.assertIn("S39_BATCH_EVIDENCE_ERROR", process.stderr)
        self.assertNotIn("Traceback", process.stderr)
        self.assertEqual(process.stdout, "")


if __name__ == "__main__":
    unittest.main()
