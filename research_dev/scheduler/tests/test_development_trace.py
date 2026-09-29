from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

from research_dev.scheduler.campaigns.burstgpt.runner import (
    GEMMA_ROLE,
    QWEN_ROLE,
    apply_named_replay_schedule,
    merge_rows,
)


REPO = Path(__file__).resolve().parents[3]
TRACES = REPO / "research_dev/scheduler/campaigns/burstgpt/data/traces"
LARGE = REPO / (
    "research_dev/spikes/s41_gemma_qwen_continuous_baseline/"
    "tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1/REQUESTS_SEMANTIC_SOURCE.jsonl"
)
OVERLAY = REPO / (
    "research_dev/spikes/s42_general_energy_scheduler_v1/"
    "full_fp16_burstgpt_v1/small_model_overlay_v1/REQUESTS_LLAMA1B_10.jsonl"
)


class DevelopmentTraceTests(unittest.TestCase):
    def test_long_variant_uses_unchanged_real_generations(self):
        parent = json.loads((TRACES / "burstgpt_sparse_locality24_v1.json").read_text())
        dev = json.loads((TRACES / "burstgpt_dev3_long_v1.json").read_text())
        large = [json.loads(row) for row in LARGE.read_text().splitlines() if row]
        overlay = [json.loads(row) for row in OVERLAY.read_text().splitlines() if row]
        merged = merge_rows(large, overlay, {
            QWEN_ROLE: "qwen-test-artifact", GEMMA_ROLE: "gemma-test-artifact",
        })
        before = copy.deepcopy(merged)
        selected, summary = apply_named_replay_schedule(merged, dev)
        self.assertEqual((selected, summary), apply_named_replay_schedule(merged, dev))
        self.assertEqual(merged, before)
        indices = [row["combined_index"] for row in selected]
        self.assertEqual(indices, [36, 37, 50])
        self.assertEqual(indices, [
            row["combined_request_index"] for row in parent["arrivals"]
            if row["combined_request_index"] in indices
        ])
        self.assertEqual(summary["trace_name"], "burstgpt_dev3_long_v1")
        self.assertEqual(summary["replay_span_us"], 90_000_000)
        self.assertEqual(
            [row["row"]["arrival_us"] for row in selected],
            [1_000_000, 61_000_000, 91_000_000],
        )
        self.assertEqual(
            [(row["row"]["input_tokens"], row["row"]["output_tokens"])
             for row in selected],
            [(915, 292), (915, 292), (277, 71)],
        )
        self.assertEqual(sum(row["row"]["output_tokens"] for row in selected), 655)
        self.assertEqual(len({row["model_id"] for row in selected}), 3)
        for row in selected:
            expected = copy.deepcopy(before[row["combined_index"]])
            expected["row"].update(
                arrival_us=row["row"]["arrival_us"],
                replay_arrival_us=row["row"]["arrival_us"],
                source_arrival_us=expected["row"]["arrival_us"],
            )
            self.assertEqual(row, expected)

    def test_ordered_subset_and_parent_identity(self):
        parent_path = TRACES / "burstgpt_sparse_locality24_v1.json"
        self.assertEqual(
            hashlib.sha256(parent_path.read_bytes()).hexdigest(),
            "78b7582ebfe063cdbb14added28b060ee955362d907cdfc1b5b7c6aa14192ce9",
        )
        parent = json.loads(parent_path.read_text())
        dev = json.loads((TRACES / "burstgpt_dev4_5min_v1.json").read_text())
        indices = [row["combined_request_index"] for row in dev["arrivals"]]
        self.assertEqual(indices, [34, 40, 56, 57])
        self.assertEqual(indices, [
            row["combined_request_index"] for row in parent["arrivals"]
            if row["combined_request_index"] in indices
        ])
        self.assertEqual(
            [row["replay_arrival_us"] for row in dev["arrivals"]],
            [1_000_000, 31_000_000, 71_000_000, 91_000_000],
        )

    def test_real_work_preserved_and_replay_deterministic(self):
        large = [json.loads(row) for row in LARGE.read_text().splitlines() if row]
        overlay = [json.loads(row) for row in OVERLAY.read_text().splitlines() if row]
        merged = merge_rows(large, overlay, {
            QWEN_ROLE: "qwen-test-artifact", GEMMA_ROLE: "gemma-test-artifact",
        })
        before = copy.deepcopy(merged)
        dev = json.loads((TRACES / "burstgpt_dev4_5min_v1.json").read_text())
        selected, summary = apply_named_replay_schedule(merged, dev)
        self.assertEqual((selected, summary), apply_named_replay_schedule(merged, dev))
        self.assertEqual(merged, before)
        self.assertEqual(summary["trace_name"], "burstgpt_dev4_5min_v1")
        self.assertEqual(summary["replay_span_us"], 90_000_000)
        self.assertEqual(
            [(row["row"]["input_tokens"], row["row"]["output_tokens"])
             for row in selected],
            [(147, 36), (178, 43), (625, 12), (271, 41)],
        )
        self.assertEqual(len({row["model_id"] for row in selected}), 3)
        for row in selected:
            original = before[row["combined_index"]]
            self.assertEqual(
                {key: value for key, value in row.items() if key != "row"},
                {key: value for key, value in original.items() if key != "row"},
            )
            expected = dict(original["row"])
            expected.update(
                arrival_us=row["row"]["arrival_us"],
                replay_arrival_us=row["row"]["arrival_us"],
                source_arrival_us=original["row"]["arrival_us"],
            )
            self.assertEqual(row["row"], expected)


if __name__ == "__main__":
    unittest.main()
