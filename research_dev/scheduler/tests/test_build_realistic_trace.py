"""Pure parts of the realistic-trace builder: merge-order indices and window selection."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from research_dev.scheduler.campaigns.burstgpt.build_realistic_trace import (
    same_model_overlaps,
    assign_combined_indices, model_inventory, parse_execution_artifact, select_window,
)


class BuildRealisticTraceTests(unittest.TestCase):
    def test_combined_indices_follow_the_loader_merge_order(self):
        large = [{"arrival_us": 1000, "request_index": 0}, {"arrival_us": 3000, "request_index": 1},
                 {"arrival_us": 3000, "request_index": 2}]
        overlay = [{"arrival_us": 2000, "overlay_request_index": 0}, {"arrival_us": 3000, "overlay_request_index": 1}]
        # same arrival: large rows first, then overlay, each by source index (trace_inputs.merge_rows)
        self.assertEqual(assign_combined_indices(large, overlay),
                         [("large", 0, 0), ("overlay", 0, 1), ("large", 1, 2), ("large", 2, 3), ("overlay", 1, 4)])

    def test_window_selection_scans_for_the_request_count_range(self):
        rows = [{"t": float(t), "model": "ChatGPT", "input": 100, "output": 50} for t in range(0, 3600, 60)]
        window, info = select_window(rows, duration_s=600, min_requests=8, max_requests=12, start_offset_s=None,
                                     min_input=1, min_output=1)
        self.assertEqual(len(window), 10)
        self.assertEqual(info["window_start_source_s"], 0.0)
        with self.assertRaises(SystemExit):
            select_window(rows, duration_s=600, min_requests=50, max_requests=60, start_offset_s=None,
                          min_input=1, min_output=1)

    def test_long_tail_rule_takes_the_first_window_matching_the_log_shares(self):
        # six 600 s windows of ten rows; long outputs (1000) per window: 0, 10, 2, 2, 2, 2
        # log share = 18/60 = 0.30 of requests; window 0 has none, window 1 is all long,
        # windows 2..5 have 0.2 and match at tolerance 0.15 -- the first of them must be taken
        long_per_window = [0, 10, 2, 2, 2, 2]
        rows = []
        for w, n_long in enumerate(long_per_window):
            for i in range(10):
                rows.append({"t": float(w * 600 + i * 60), "model": "ChatGPT", "input": 100,
                             "output": 1000 if i < n_long else 100})
        window, info = select_window(rows, duration_s=600, min_requests=8, max_requests=12, start_offset_s=None,
                                     min_input=1, min_output=1, long_tail_threshold=512, long_tail_tolerance=0.15)
        self.assertEqual(info["window_start_source_s"], 1200.0)
        self.assertAlmostEqual(info["long_tail"]["log"]["request_share"], 0.30)
        self.assertEqual(info["long_tail"]["window"]["long_requests"], 2)
        self.assertEqual(sum(r["output"] > 512 for r in window), 2)
        with self.assertRaises(SystemExit):
            select_window(rows, duration_s=600, min_requests=8, max_requests=12, start_offset_s=None,
                          min_input=1, min_output=1, long_tail_threshold=512, long_tail_tolerance=0.01)

    def test_max_output_tokens_skips_windows_that_would_run_too_long(self):
        # three 600 s windows of ten rows (the scan needs a full window after each start); the
        # first carries 5000 output tokens, the others 1000
        rows = [{"t": float(w * 600 + i * 60), "model": "ChatGPT", "input": 100,
                 "output": 500 if w == 0 else 100} for w in range(3) for i in range(10)]
        window, info = select_window(rows, duration_s=600, min_requests=8, max_requests=12, start_offset_s=None,
                                     min_input=1, min_output=1, max_output_tokens=1500, output_cap=400)
        self.assertEqual(info["window_start_source_s"], 600.0)
        self.assertEqual(info["max_output_tokens"], 1500)
        self.assertEqual(sum(r["output"] for r in window), 1000)
        # the cap is applied before the bound: 10 x min(500, 100) = 1000 passes with output_cap 100
        window, info = select_window(rows, duration_s=600, min_requests=8, max_requests=12, start_offset_s=None,
                                     min_input=1, min_output=1, max_output_tokens=1500, output_cap=100)
        self.assertEqual(info["window_start_source_s"], 0.0)
        with self.assertRaises(SystemExit):
            select_window(rows, duration_s=600, min_requests=8, max_requests=12, start_offset_s=None,
                          min_input=1, min_output=1, max_output_tokens=500, output_cap=None)

    def test_same_model_overlaps_counts_pairs_that_can_share_a_batch(self):
        rows = [{"t": 0.0, "model": "ChatGPT", "input": 10, "output": 100},   # alone: done at 50 s
                {"t": 40.0, "model": "ChatGPT", "input": 10, "output": 10},   # arrives before 50 s -> overlap
                {"t": 45.0, "model": "GPT-4", "input": 10, "output": 10},     # other model -> no pair
                {"t": 120.0, "model": "ChatGPT", "input": 10, "output": 10}]  # after both finish
        self.assertEqual(same_model_overlaps(rows, arrival_scale=1.0, output_cap=None, service_s_per_token=0.5), 1)
        # compressing arrivals 4x puts the last request (30 s) inside the first one's 50 s as well
        self.assertEqual(same_model_overlaps(rows, arrival_scale=0.25, output_cap=None, service_s_per_token=0.5), 2)
        # capping the first output at 20 tokens (10 s) removes both overlaps at scale 1
        self.assertEqual(same_model_overlaps(rows, arrival_scale=1.0, output_cap=20, service_s_per_token=0.5), 0)

    def test_concurrency_rule_skips_windows_without_overlap_or_model_mix(self):
        # window 0: ten spaced single-model requests (no overlap); window 1: same-model bursts, both models
        rows = [{"t": float(i * 60), "model": "ChatGPT", "input": 100, "output": 20} for i in range(10)]
        rows += [{"t": 600.0 + i * 5, "model": "ChatGPT" if i % 2 else "GPT-4", "input": 100, "output": 200}
                 for i in range(10)]
        rows += [{"t": 1200.0 + i * 60, "model": "ChatGPT", "input": 100, "output": 20} for i in range(10)]
        window, info = select_window(rows, duration_s=600, min_requests=8, max_requests=12, start_offset_s=None,
                                     min_input=1, min_output=1, min_same_model_overlaps=4,
                                     min_requests_per_model=2, arrival_scale=1.0, service_s_per_token=0.5)
        self.assertEqual(info["window_start_source_s"], 600.0)
        self.assertGreaterEqual(info["concurrency"]["same_model_overlaps"], 4)
        with self.assertRaises(SystemExit):
            select_window(rows, duration_s=600, min_requests=8, max_requests=12, start_offset_s=None,
                          min_input=1, min_output=1, min_same_model_overlaps=1000, arrival_scale=1.0)

    def test_model_inventory_pins_bytes_and_sha256_of_each_execution_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tiny.gguf"
            path.write_bytes(b"gguf-bytes")
            spec = parse_execution_artifact(f"tiny-model={path}=text_decoder_f16_proxy")
            inventory = model_inventory([spec, parse_execution_artifact(f"other={path}")])
        self.assertEqual(inventory["tiny-model"], {
            "artifact_bytes": 10, "artifact_file": "tiny.gguf",
            "artifact_sha256": hashlib.sha256(b"gguf-bytes").hexdigest(), "kind": "text_decoder_f16_proxy"})
        self.assertEqual(inventory["other"]["kind"], "text_decoder")
        with self.assertRaises(ValueError):
            model_inventory([spec, spec])

    def test_model_inventory_rejects_trace_role_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tiny.gguf"
            path.write_bytes(b"gguf-bytes")
            with self.assertRaises(SystemExit):
                model_inventory([parse_execution_artifact(f"qwen3-14b-q4_k_m={path}")])


if __name__ == "__main__":
    unittest.main()
