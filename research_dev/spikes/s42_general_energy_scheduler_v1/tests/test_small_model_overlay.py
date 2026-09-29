#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
OVERLAY_ROOT = ROOT / "small_model_overlay_v1"
sys.path[:0] = [str(REPO_ROOT), str(OVERLAY_ROOT)]

from build_small_model_overlay import (  # noqa: E402
    MANIFEST,
    OVERLAY_MODEL,
    OVERLAY_STREAM,
    TRACE,
)
from evaluate_small_model_placement import (  # noqa: E402
    OUTPUT,
    canonical,
    evaluate,
)
from verify_small_model_overlay import (  # noqa: E402
    TraceValidationError,
    validate,
)
from research_dev.scheduler import (  # noqa: E402
    SMALL_MODEL_OVERLAY_TRACE_SCHEMA,
    load_trace,
)


class SmallModelOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = [
            json.loads(line)
            for line in TRACE.read_text(encoding="ascii").splitlines()
        ]
        cls.manifest = json.loads(MANIFEST.read_text(encoding="ascii"))

    def test_checked_in_derivation_passes(self) -> None:
        result = validate()
        self.assertEqual(result["records"], 84)
        self.assertEqual(
            result["models"],
            {
                "gemma-4-12b-it-q4_0": 17,
                "llama-3.2-1b-instruct-q4_0": 10,
                "qwen3-14b-q4_k_m": 57,
            },
        )

    def test_trace_adapter_accepts_focused_schema(self) -> None:
        workload_map = {
            model_id: model_id for model_id in self.manifest["models"]
        }
        trace = load_trace(TRACE, workload_map, "bounded_numeric")
        self.assertEqual(len(trace.requests), 84)
        self.assertEqual(
            Counter(request.workload_id for request in trace.requests),
            Counter(self.manifest["models"]),
        )
        self.assertTrue(all(
            row["schema"] == SMALL_MODEL_OVERLAY_TRACE_SCHEMA
            for row in self.rows
        ))

    def test_small_stream_preserves_matched_shapes(self) -> None:
        originals = {
            row["event_id"]: row
            for row in self.rows
            if row["trace_stream_id"] == "burstgpt-original"
        }
        overlay = [
            row
            for row in self.rows
            if row["trace_stream_id"] == OVERLAY_STREAM
        ]
        self.assertEqual(len(overlay), 10)
        self.assertTrue(all(
            row["execution_model_id"] == OVERLAY_MODEL for row in overlay
        ))
        for row in overlay:
            donor = originals[row["source_shape_event_id"]]
            self.assertEqual(row["arrival_us"], donor["arrival_us"] + 200_000)
            self.assertEqual(row["input_tokens"], donor["input_tokens"])
            self.assertEqual(row["output_tokens"], donor["output_tokens"])
            self.assertEqual(row["slo_us"], donor["slo_us"])

    def test_state_and_shape_placement_expectations(self) -> None:
        result = evaluate()
        routes = {
            name: case["routes"] for name, case in result["cases"].items()
        }
        self.assertEqual(
            routes["cuda_epoch_open_all_resident"],
            {"phone-adreno": list(range(10))},
        )
        self.assertEqual(
            routes["cuda_epoch_reused_all_resident"],
            {
                "desktop-cuda": [0, 1, 2, 3, 4, 7, 8, 9],
                "phone-adreno": [5, 6],
            },
        )
        self.assertEqual(
            routes["cuda_busy_phone_resident"],
            {"phone-adreno": list(range(10))},
        )
        self.assertEqual(
            routes["current_saturated_no_transition"],
            {"desktop-cpu": list(range(10))},
        )
        self.assertEqual(OUTPUT.read_bytes(), canonical(result))

    def test_saturated_snapshot_does_not_assume_accelerator_fit(self) -> None:
        result = evaluate()
        capacity = result["saturated_snapshot"]
        self.assertEqual(
            capacity["gpu"]["stageable_shortfall_bytes"], 117_665_440
        )
        self.assertEqual(
            capacity["phone"]["stageable_shortfall_bytes"], 522_759_840
        )

    def test_trace_mutation_is_rejected(self) -> None:
        content = bytearray(TRACE.read_bytes())
        marker = b'"output_tokens":192'
        offset = content.find(marker)
        self.assertGreaterEqual(offset, 0)
        content[offset : offset + len(marker)] = b'"output_tokens":193'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_bytes(content)
            with self.assertRaises(TraceValidationError):
                validate(path, MANIFEST)


if __name__ == "__main__":
    unittest.main()
