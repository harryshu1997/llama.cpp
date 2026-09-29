#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIXED_ROOT = ROOT / "mixed_model_trace_v1"
if str(MIXED_ROOT) not in sys.path:
    sys.path.insert(0, str(MIXED_ROOT))

from verify_mixed_trace import (  # noqa: E402
    MANIFEST,
    TRACE,
    TraceValidationError,
    canonical,
    validate,
)


class MixedModelTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = [json.loads(raw) for raw in TRACE.read_text().splitlines()]
        cls.manifest = json.loads(MANIFEST.read_text())

    def write_case(
        self,
        rows: list[dict[str, object]],
        manifest: dict[str, object],
    ) -> tuple[Path, Path]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        trace_path = root / "trace.jsonl"
        manifest_path = root / "manifest.json"
        trace_content = b"".join(canonical(row) for row in rows)
        trace_meta = manifest["trace"]
        assert isinstance(trace_meta, dict)
        trace_meta["sha256"] = hashlib.sha256(trace_content).hexdigest()
        trace_path.write_bytes(trace_content)
        manifest_path.write_bytes(canonical(manifest))
        return trace_path, manifest_path

    def test_checked_in_trace_passes(self) -> None:
        result = validate()
        self.assertEqual(result["records"], 114)
        self.assertEqual(len(result["models"]), 6)

    def test_parent_field_mutation_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        manifest = copy.deepcopy(self.manifest)
        original = next(
            row for row in rows if row["trace_stream_id"] == "burstgpt-original"
        )
        original["source_model"] = "changed"
        trace_path, manifest_path = self.write_case(rows, manifest)
        with self.assertRaises(TraceValidationError):
            validate(trace_path, manifest_path)

    def test_overlay_geometry_mutation_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        manifest = copy.deepcopy(self.manifest)
        overlay = next(
            row for row in rows if row["trace_stream_id"] == "qwen3-8b-overlay"
        )
        overlay["output_tokens"] += 1
        manifest["output_tokens"] += 1
        trace_path, manifest_path = self.write_case(rows, manifest)
        with self.assertRaises(TraceValidationError):
            validate(trace_path, manifest_path)

    def test_duplicate_event_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        manifest = copy.deepcopy(self.manifest)
        rows[1]["event_id"] = rows[0]["event_id"]
        trace_path, manifest_path = self.write_case(rows, manifest)
        with self.assertRaises(TraceValidationError):
            validate(trace_path, manifest_path)

    def test_image_identity_mutation_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        manifest = copy.deepcopy(self.manifest)
        image_row = next(row for row in rows if row["modality"] == "image_text")
        image_row["image"]["sha256"] = "0" * 64
        trace_path, manifest_path = self.write_case(rows, manifest)
        with self.assertRaises(TraceValidationError):
            validate(trace_path, manifest_path)

    def test_unmeasured_image_tokens_are_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        manifest = copy.deepcopy(self.manifest)
        image_row = next(row for row in rows if row["modality"] == "image_text")
        image_row["image_tokens"] = 256
        trace_path, manifest_path = self.write_case(rows, manifest)
        with self.assertRaises(TraceValidationError):
            validate(trace_path, manifest_path)

    def test_text_prompt_transport_mutation_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        manifest = copy.deepcopy(self.manifest)
        text_row = next(
            row for row in rows if row["trace_stream_id"] == "qwen3-0.6b-overlay"
        )
        text_row["prompt_transport"] = "multimodal_message"
        trace_path, manifest_path = self.write_case(rows, manifest)
        with self.assertRaises(TraceValidationError):
            validate(trace_path, manifest_path)

    def test_vlm_raw_prompt_mutation_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        manifest = copy.deepcopy(self.manifest)
        image_row = next(row for row in rows if row["modality"] == "image_text")
        image_row["prompt_text"] = "changed"
        trace_path, manifest_path = self.write_case(rows, manifest)
        with self.assertRaises(TraceValidationError):
            validate(trace_path, manifest_path)

    def test_manifest_aggregate_mutation_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        manifest = copy.deepcopy(self.manifest)
        manifest["output_tokens"] += 1
        trace_path, manifest_path = self.write_case(rows, manifest)
        with self.assertRaises(TraceValidationError):
            validate(trace_path, manifest_path)


if __name__ == "__main__":
    unittest.main()
