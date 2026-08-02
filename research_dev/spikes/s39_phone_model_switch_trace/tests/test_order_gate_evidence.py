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
SCRIPT = S39 / "summarize_order_gate.py"
EVIDENCE = S39 / "results" / "w3_order_gate"

SPEC = importlib.util.spec_from_file_location("s39_order_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class OrderGateEvidenceTests(unittest.TestCase):
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

    def test_current_evidence_passes(self):
        result = SUMMARY.build()
        self.assertEqual(result["status"], "ROW_ORDER_EFFECT_PASS")
        self.assertGreaterEqual(
            result["gate"]["aggregate_speedup_milli"],
            SUMMARY.SPEEDUP_GATE_MILLI,
        )
        self.assertEqual(result["correctness"]["token_checks"], 1024)
        self.assertTrue(result["gate"]["thermal_matched"])

    def test_checked_in_certificate_matches_reducer(self):
        self.assertEqual(
            (EVIDENCE / "order_gate_certificate.json").read_bytes(),
            SUMMARY.common.canonical_bytes(SUMMARY.build()),
        )

    def test_permutation_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_order_") as directory:
            evidence = self.copy_evidence(Path(directory))
            self.rewrite_json(
                evidence / "shuffled_ab.json",
                lambda value: value["permutation"].__setitem__(0, 0),
            )
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "permutation",
            ):
                SUMMARY.build(evidence)

    def test_compute_total_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_order_") as directory:
            evidence = self.copy_evidence(Path(directory))
            self.rewrite_json(
                evidence / "sorted_ab.json",
                lambda value: value["batch_summary"].__setitem__(
                    "compute_us_total",
                    1,
                ),
            )
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "compute_us_total",
            ):
                SUMMARY.build(evidence)

    def test_worker_fallback_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_order_") as directory:
            evidence = self.copy_evidence(Path(directory))
            path = evidence / "op12_ab.log"
            raw = path.read_bytes()
            mutated = raw.replace(
                b'"ADD":{"OpenCL":160}',
                b'"ADD":{"CPU":160}',
                1,
            )
            self.assertNotEqual(raw, mutated)
            path.write_bytes(mutated)
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "CPU fallback",
            ):
                SUMMARY.build(evidence)

    def test_relay_row_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_order_") as directory:
            evidence = self.copy_evidence(Path(directory))
            path = evidence / "relay_sorted_ab.log"
            raw = path.read_bytes()
            mutated = raw.replace(b'"rows":384', b'"rows":385', 1)
            self.assertNotEqual(raw, mutated)
            path.write_bytes(mutated)
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "relay.rows",
            ):
                SUMMARY.build(evidence)

    def test_thermal_boot_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_order_") as directory:
            evidence = self.copy_evidence(Path(directory))
            self.rewrite_json(
                evidence / "thermal_sorted_ab_before.json",
                lambda value: value["samples"]["OP15"].__setitem__(
                    "device_boot_id",
                    "other-boot",
                ),
            )
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "device_boot_id",
            ):
                SUMMARY.build(evidence)

    def test_runtime_binary_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_order_") as directory:
            evidence = self.copy_evidence(Path(directory))
            self.rewrite_json(
                evidence / "RUN_CONTEXT.json",
                lambda value: value["op15"].__setitem__(
                    "layersplit_sha256",
                    "0" * 64,
                ),
            )
            with self.assertRaisesRegex(
                SUMMARY.common.BatchEvidenceError,
                "run_context.op15",
            ):
                SUMMARY.build(evidence)

    def test_thermal_imbalance_downgrades_status(self):
        with tempfile.TemporaryDirectory(prefix="s39_order_") as directory:
            evidence = self.copy_evidence(Path(directory))

            def heat(value):
                sample = value["samples"]["OP15"]
                for zone in sample["gpu_zones_millic"]:
                    sample["gpu_zones_millic"][zone] += 10000
                sample["gpu_max_millic"] += 10000

            self.rewrite_json(
                evidence / "thermal_sorted_ab_before.json",
                heat,
            )
            result = SUMMARY.build(evidence)
            self.assertEqual(
                result["status"],
                "ROW_ORDER_EFFECT_THERMAL_UNMATCHED",
            )
            self.assertFalse(result["gate"]["thermal_matched"])

    def test_cli_fails_closed_without_traceback(self):
        with tempfile.TemporaryDirectory(prefix="s39_order_") as directory:
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
        self.assertIn("S39_ORDER_GATE_EVIDENCE_ERROR", process.stderr)
        self.assertNotIn("Traceback", process.stderr)
        self.assertEqual(process.stdout, "")


if __name__ == "__main__":
    unittest.main()
