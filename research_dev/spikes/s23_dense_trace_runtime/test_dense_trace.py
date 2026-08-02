#!/usr/bin/env python3

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dense_trace import TraceError, build, content_digest, validate


class DenseTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value = build()

    def test_frozen_dense_window(self) -> None:
        self.assertEqual(self.value["selection"]["densest_window_count"], 60)
        self.assertEqual(self.value["selection"]["selected_count"], 60)
        self.assertEqual(self.value["observed_summary"]["arrival_span_us"], 2_000_000)
        validate(self.value)

    def test_source_demand_is_not_relabelled_as_executed_payload(self) -> None:
        self.assertEqual(
            self.value["execution_proxy"]["claims"],
            "arrival-pressure-and-runtime-mechanics-only",
        )
        self.assertGreater(self.value["observed_summary"]["input_tokens_max"], 512)
        self.assertTrue(all(row["execution_input_tokens"] == 1 for row in self.value["requests"]))

    def test_digest_mutation_fails(self) -> None:
        mutated = copy.deepcopy(self.value)
        mutated["requests"][0]["arrival_us"] += 1
        with self.assertRaisesRegex(TraceError, "content digest"):
            validate(mutated, verify_source=False)

    def test_rehashed_class_mutation_still_fails(self) -> None:
        mutated = copy.deepcopy(self.value)
        mutated["requests"][0]["priority"] = (mutated["requests"][0]["priority"] + 1) % 3
        mutated["trace_hash"] = content_digest(mutated)
        with self.assertRaisesRegex(TraceError, "class binding"):
            validate(mutated, verify_source=False)

    def test_duplicate_request_fails(self) -> None:
        mutated = copy.deepcopy(self.value)
        mutated["requests"][1]["request_id"] = mutated["requests"][0]["request_id"]
        mutated["requests"][1]["source_row_id"] = mutated["requests"][0]["source_row_id"]
        mutated["trace_hash"] = content_digest(mutated)
        with self.assertRaisesRegex(TraceError, "identity"):
            validate(mutated, verify_source=False)

    def test_rehashed_summary_mutation_fails_without_source(self) -> None:
        mutated = copy.deepcopy(self.value)
        mutated["observed_summary"]["input_tokens_total"] += 1
        mutated["trace_hash"] = content_digest(mutated)
        with self.assertRaisesRegex(TraceError, "summary"):
            validate(mutated, verify_source=False)

    def test_bool_proxy_shape_fails(self) -> None:
        mutated = copy.deepcopy(self.value)
        mutated["requests"][0]["execution_input_tokens"] = True
        mutated["trace_hash"] = content_digest(mutated)
        with self.assertRaisesRegex(TraceError, "integer"):
            validate(mutated, verify_source=False)


if __name__ == "__main__":
    unittest.main()
