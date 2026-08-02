#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_adapter import TraceError, build_trace, canonical_bytes  # noqa: E402


SOURCE = REPO / "research_dev/spikes/s23_dense_trace_runtime/burstgpt-dense-60.json"


class TraceAdapterTests(unittest.TestCase):
    def test_real_source_is_deterministic_and_bound(self) -> None:
        first = build_trace(SOURCE)
        second = build_trace(SOURCE)
        self.assertEqual(canonical_bytes(first), canonical_bytes(second))
        self.assertEqual(len(first["requests"]), 60)
        self.assertTrue(first["source"]["file_sha256"].startswith("sha256:"))
        self.assertEqual(first["execution_proxy"]["prompt_tokens"], 4)
        self.assertTrue(all(
            row["prompt_tokens"] == [2, 2, 2, 2]
            and row["output_steps"] == 4
            for row in first["requests"]
        ))

    def test_duplicate_request_is_rejected(self) -> None:
        value = json.loads(SOURCE.read_text(encoding="utf-8"))
        value["requests"][1]["request_id"] = value["requests"][0]["request_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(TraceError):
                build_trace(path)

    def test_wrong_execution_shape_is_rejected(self) -> None:
        value = json.loads(SOURCE.read_text(encoding="utf-8"))
        value["requests"][0]["execution_steps"] = 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(TraceError):
                build_trace(path)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            with self.assertRaises(TraceError):
                build_trace(path)


if __name__ == "__main__":
    unittest.main()
