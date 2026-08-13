#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "research_dev/spikes/s42_general_energy_scheduler_v1"
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    PROFILE_SCHEMA,
    ProfileBundle,
    SOURCE_TRACE_SCHEMA as SOURCE_SCHEMA,
    TraceError,
    UnifiedScheduler,
    load_burstgpt_trace as load_burstgpt,
    load_mixed_model_trace,
    load_trace,
)


TWO_MODEL_TRACE = (
    REPO_ROOT
    / "research_dev/spikes/s41_gemma_qwen_continuous_baseline"
    / "tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1"
    / "REQUESTS_SEMANTIC_SOURCE.jsonl"
)
SIX_MODEL_TRACE = ROOT / "mixed_model_trace_v1/REQUESTS_MIXED_114.jsonl"


def trace_profile(model_ids: set[str]) -> ProfileBundle:
    routes = []
    for model_id in sorted(model_ids):
        routes.append({
            "route_id": f"{model_id}-baseline",
            "workload_id": model_id,
            "granularity": "task",
            "baseline": True,
            "resource_slots": {"compute": 1},
            "latency": {
                "cost_us": {
                    "kind": "affine_tokens_v1",
                    "fixed": 1,
                    "input_token": 0,
                    "output_token": 0,
                },
                "ucb_add_us": 0,
                "sample_count": 1,
                "measured": True,
            },
            "energy": {
                "status": "unknown",
                "cost_uj": None,
                "lower_error_ppm": 0,
                "upper_error_ppm": 0,
            },
            "overlap": {"status": "not_applicable"},
            "quality_class": "exact",
            "placement_verified": True,
            "resident": True,
            "server_busy_ppm": 1_000_000,
            "server_memory_bytes": 1,
            "evidence_ids": ["test-only-cross-trace-profile"],
        })
    return ProfileBundle.from_json({
        "schema": PROFILE_SCHEMA,
        "profile_id": "test-only-cross-trace-profile",
        "resources": [{
            "resource_id": "compute",
            "kind": "cpu",
            "capacity": 1,
            "ready": True,
            "identity": "test-compute",
        }],
        "trace_workload_map": {
            model_id: model_id for model_id in sorted(model_ids)
        },
        "policy": {},
        "routes": routes,
    })


def row(event_id: str, arrival_us: int = 10) -> dict[str, object]:
    return {
        "schema": SOURCE_SCHEMA,
        "event_id": event_id,
        "model_id": "model",
        "arrival_us": arrival_us,
        "slo_us": 1000,
        "input_tokens": 12,
        "output_tokens": 4,
    }


class TraceTests(unittest.TestCase):
    def write_rows(self, rows: list[dict[str, object]]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trace.jsonl"
        path.write_text(
            "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in rows),
            encoding="ascii",
        )
        return path

    def test_preserves_arrivals_shapes_and_deadlines(self) -> None:
        path = self.write_rows([row("a", 10), row("b", 20)])
        trace = load_burstgpt(path, {"model": "work"})
        self.assertEqual(len(trace.requests), 2)
        self.assertEqual(trace.requests[0].arrival_us, 10)
        self.assertEqual(trace.requests[0].deadline_us, 1010)
        self.assertEqual(trace.requests[0].input_tokens, 12)
        self.assertEqual(trace.output_tokens, 8)

    def test_unknown_model_is_rejected(self) -> None:
        path = self.write_rows([row("a")])
        with self.assertRaises(TraceError):
            load_burstgpt(path, {"other": "work"})

    def test_duplicate_event_is_rejected(self) -> None:
        path = self.write_rows([row("a", 10), row("a", 20)])
        with self.assertRaises(TraceError):
            load_burstgpt(path, {"model": "work"})

    def test_out_of_order_arrival_is_rejected(self) -> None:
        path = self.write_rows([row("a", 20), row("b", 10)])
        with self.assertRaises(TraceError):
            load_burstgpt(path, {"model": "work"})

    def test_duplicate_json_key_is_rejected(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trace.jsonl"
        path.write_text(
            '{"schema":"%s","schema":"%s"}\n' % (SOURCE_SCHEMA, SOURCE_SCHEMA),
            encoding="ascii",
        )
        with self.assertRaises(TraceError):
            load_burstgpt(path, {"model": "work"})

    def test_mixed_trace_preserves_all_models_and_features(self) -> None:
        trace_path = SIX_MODEL_TRACE
        model_ids = {
            json.loads(line)["execution_model_id"]
            for line in trace_path.read_text(encoding="ascii").splitlines()
        }
        workload_map = {model_id: model_id for model_id in model_ids}
        trace = load_mixed_model_trace(trace_path, workload_map)
        self.assertEqual(len(trace.requests), 114)
        self.assertEqual(trace.output_tokens, 16_506)
        self.assertEqual(
            {request.workload_id for request in trace.requests},
            model_ids,
        )
        vlm = [
            request
            for request in trace.requests
            if request.features["is_multimodal"] == 1
        ]
        self.assertEqual(len(vlm), 10)
        self.assertTrue(all(request.features["image_bytes"] > 0 for request in vlm))

    def test_schema_dispatches_mixed_trace(self) -> None:
        trace_path = SIX_MODEL_TRACE
        model_ids = {
            json.loads(line)["execution_model_id"]
            for line in trace_path.read_text(encoding="ascii").splitlines()
        }
        trace = load_trace(
            trace_path,
            {model_id: model_id for model_id in model_ids},
        )
        self.assertEqual(len(trace.requests), 114)

    def test_one_scheduler_api_replays_two_and_six_model_traces(self) -> None:
        cases = (
            (
                TWO_MODEL_TRACE,
                "model_id",
                Counter({
                    "gemma-4-12b-it-q8_0": 57,
                    "qwen3-14b-q4_k_m": 17,
                }),
            ),
            (
                SIX_MODEL_TRACE,
                "execution_model_id",
                Counter({
                    "gemma-4-12b-it-q4_0": 17,
                    "gemma-4-e2b-it-q8_0-vlm": 10,
                    "llama-3.2-1b-instruct-q4_0": 10,
                    "qwen3-0.6b-q8_0": 10,
                    "qwen3-14b-q4_k_m": 57,
                    "qwen3-8b-q8_0": 10,
                }),
            ),
        )
        all_models = set().union(*(set(expected) for _, _, expected in cases))
        profile = trace_profile(all_models)

        for path, model_key, expected in cases:
            with self.subTest(path=path.name):
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="ascii").splitlines()
                ]
                workload_map = {
                    row[model_key]: row[model_key] for row in rows
                }
                trace = load_trace(path, workload_map)
                scheduler = UnifiedScheduler((profile,), "control")
                decisions = []
                for request in trace.requests:
                    decision = scheduler.schedule(request)
                    scheduler.release_decision(decision, decision.finish_us)
                    decisions.append(decision)

                self.assertEqual(len(decisions), sum(expected.values()))
                self.assertEqual(
                    Counter(request.workload_id for request in trace.requests),
                    expected,
                )
                self.assertEqual(
                    Counter(decision.route_id for decision in decisions),
                    Counter({
                        f"{model_id}-baseline": count
                        for model_id, count in expected.items()
                    }),
                )


if __name__ == "__main__":
    unittest.main()
