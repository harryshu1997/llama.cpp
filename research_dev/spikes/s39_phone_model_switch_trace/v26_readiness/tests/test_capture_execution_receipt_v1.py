#!/usr/bin/env python3

import base64
import json
from pathlib import Path
import tempfile
import unittest
import sys


HERE = Path(__file__).resolve().parents[1]
import importlib.util

spec = importlib.util.spec_from_file_location("receipt", HERE / "capture_execution_receipt_v1.py")
receipt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receipt)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "producer.py"
        self.output = self.root / "result.json"
        self.input = self.root / "input.json"
        self.input.write_bytes(b'{"input":1}\n')
        self.source.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "Path(sys.argv[1]).write_bytes(b'{\\\"result\\\":1}\\n')\n",
            encoding="ascii",
        )

    def tearDown(self):
        self.temp.cleanup()

    def run_receipt(self):
        return receipt.execute(
            capture_kind="artifact_root",
            producer_role="capture.artifact_root",
            phase_id="cp0-r1-v26-a-only-test",
            contract_sha256="a" * 64,
            execution_plan_sha256="b" * 64,
            runtime_bundle_plan_sha256="c" * 64,
            source_path=self.source,
            input_paths=[("python", Path(sys.executable).resolve())]
            + [(f"input-{index}", self.input) for index in range(4)],
            producer_argv=[str(self.source), str(self.output)],
            cwd=self.root,
            environment={"PATH": "/usr/bin:/bin"},
            timeout_seconds=10,
            result_path=self.output,
        )

    def test_fd_execution_and_revalidation(self):
        value = self.run_receipt()
        receipt.validate_receipt(value, source_path=self.source, result_path=self.output)
        self.assertEqual(value["process"]["returncode"], 0)
        self.assertEqual(value["source"]["execution_mode"], "VERIFIED_OPEN_FD")

    def test_source_mutation_after_receipt_is_rejected(self):
        value = self.run_receipt()
        self.source.write_bytes(self.source.read_bytes() + b"\n")
        with self.assertRaisesRegex(receipt.ReceiptError, "E_SOURCE"):
            receipt.validate_receipt(value, source_path=self.source, result_path=self.output)

    def test_nonzero_producer_is_rejected(self):
        self.source.write_text("raise SystemExit(4)\n", encoding="ascii")
        with self.assertRaisesRegex(receipt.ReceiptError, "E_RETURNCODE"):
            self.run_receipt()

    def test_timeout_is_rejected(self):
        self.source.write_text("import time\ntime.sleep(2)\n", encoding="ascii")
        with self.assertRaisesRegex(receipt.ReceiptError, "E_TIMEOUT"):
            receipt.execute(
                capture_kind="fast_fresh_readiness",
                producer_role="capture.fast_fresh_readiness",
                phase_id="cp0-r1-v26-a-only-test",
                contract_sha256="a" * 64,
                execution_plan_sha256="b" * 64,
                runtime_bundle_plan_sha256="c" * 64,
                source_path=self.source,
                input_paths=[("python", Path(sys.executable).resolve())]
                + [(f"input-{index}", self.input) for index in range(4)],
                producer_argv=[str(self.source), str(self.output)],
                cwd=self.root,
                environment={"PATH": "/usr/bin:/bin"},
                timeout_seconds=1,
                result_path=self.output,
            )


if __name__ == "__main__":
    unittest.main()
