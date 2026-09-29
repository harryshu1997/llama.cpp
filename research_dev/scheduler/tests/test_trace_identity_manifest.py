"""validate_trace takes the trace identity from the manifest, not from the unified trace's constants."""
import tempfile
import unittest
from pathlib import Path

from research_dev.scheduler.campaigns.burstgpt.common import UnifiedTraceError, canonical, digest
from research_dev.scheduler.campaigns.burstgpt.trace_inputs import (
    TRACE_COLD_MODEL_ID, TRACE_HOT_MODEL_ID, merge_rows, validate_trace,
)


def _row(index, role_model_id, n_prompt=8, n_output=4):
    return {
        "arrival_us": 1_000_000 + index * 1000, "event_id": f"t:{index}", "input_tokens": n_prompt,
        "model_id": role_model_id, "output_tokens": n_output, "prompt_tokenizer_model": "x",
        "prompt_tokens": list(range(n_prompt)), "request_index": index, "schema": "s43-test", "slo_us": 30_000_000,
    }


class TraceIdentityManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="s43-trace-identity-"))

    def _write(self, rows, name):
        path = self.tmp / name
        path.write_bytes(b"".join(canonical(r) for r in rows))
        return path

    def _manifest(self, large, overlay, roles, combined=None, legacy=False):
        base = {"path": str(large), "sha256": digest(large), "record_count": sum(1 for _ in large.read_bytes().splitlines())}
        if not legacy:
            base["roles"] = roles
        return {
            "schema": "s42-full-fp16-llama1b-overlay-manifest-v1",
            "base_trace": base,
            "overlay_trace": {"path": str(overlay), "sha256": digest(overlay), "record_count": sum(1 for _ in overlay.read_bytes().splitlines())},
            "combined_work": {"record_count": combined if combined is not None else base["record_count"] + sum(1 for _ in overlay.read_bytes().splitlines())},
            "model_inventory": {"model-a": {"artifact_bytes": 1, "artifact_file": "a.gguf", "artifact_sha256": "0" * 64, "kind": "text_decoder"}},
        }

    def test_manifest_without_model_inventory_is_rejected(self):
        rows = [_row(0, TRACE_HOT_MODEL_ID), _row(1, TRACE_COLD_MODEL_ID)]
        large = self._write(rows, "large.jsonl")
        overlay = self._write([], "overlay.jsonl")
        manifest = self._manifest(large, overlay, {"hot": 1, "cold": 1})
        validate_trace(large, overlay, manifest)
        del manifest["model_inventory"]
        with self.assertRaisesRegex(UnifiedTraceError, "model_inventory"):
            validate_trace(large, overlay, manifest)

    def test_manifest_with_roles_and_empty_overlay_is_accepted(self):
        rows = [_row(0, TRACE_HOT_MODEL_ID), _row(1, TRACE_COLD_MODEL_ID), _row(2, TRACE_HOT_MODEL_ID)]
        large = self._write(rows, "large.jsonl")
        overlay = self._write([], "overlay.jsonl")
        manifest = self._manifest(large, overlay, {"hot": 2, "cold": 1})
        got_large, got_overlay = validate_trace(large, overlay, manifest)
        self.assertEqual(len(got_large), 3)
        self.assertEqual(got_overlay, [])
        merged = merge_rows(got_large, got_overlay, {"hot": "qwen", "cold": "gemma"})
        self.assertEqual([m["model_id"] for m in merged], ["qwen", "gemma", "qwen"])
        self.assertEqual([m["combined_index"] for m in merged], [0, 1, 2])

    def test_role_count_mismatch_is_rejected(self):
        rows = [_row(0, TRACE_HOT_MODEL_ID), _row(1, TRACE_COLD_MODEL_ID)]
        large = self._write(rows, "large.jsonl")
        overlay = self._write([], "overlay.jsonl")
        with self.assertRaises(UnifiedTraceError):
            validate_trace(large, overlay, self._manifest(large, overlay, {"hot": 2, "cold": 0}))

    def test_record_count_mismatch_is_rejected(self):
        rows = [_row(0, TRACE_HOT_MODEL_ID)]
        large = self._write(rows, "large.jsonl")
        overlay = self._write([], "overlay.jsonl")
        manifest = self._manifest(large, overlay, {"hot": 1, "cold": 0}, combined=5)
        with self.assertRaises(UnifiedTraceError):
            validate_trace(large, overlay, manifest)

    def test_manifest_without_roles_is_only_the_legacy_74_row_trace(self):
        rows = [_row(0, TRACE_HOT_MODEL_ID)]
        large = self._write(rows, "large.jsonl")
        overlay = self._write([], "overlay.jsonl")
        with self.assertRaises(UnifiedTraceError):
            validate_trace(large, overlay, self._manifest(large, overlay, None, legacy=True))


if __name__ == "__main__":
    unittest.main()
