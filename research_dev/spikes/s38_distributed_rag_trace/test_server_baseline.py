#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare_server_baselines as compare_tool
import run_server_baseline as runner
import select_server_cohort as selector


def write_result(directory: Path, host: str, mutate: bool = False) -> None:
    rows = []
    for index in range(2):
        arrival = index * 10
        start = arrival + 5
        finish = start + 20
        response = finish - arrival + (1 if mutate and index == 1 else 0)
        rows.append({
            "schema": runner.RUN_SCHEMA,
            "event_id": f"event-{index}",
            "payload_id": f"payload-{index}",
            "arrival_due_us": arrival,
            "worker_start_us": start,
            "finish_us": finish,
            "client_queue_us": 5,
            "service_us": 20,
            "response_us": response,
            "stages_us": {"query_embedding": 2, "retrieval": 1, "reranking": 3, "generation": 14},
            "requested_input_tokens": 10,
            "requested_output_tokens": 4,
            "realized_prompt_tokens": 12,
            "realized_output_tokens": 3,
            "reasoning_budget_tokens": 2,
            "retrieved_chunk_ids": ["chunk-0"],
            "reranked_chunk_ids": ["chunk-0"],
            "generated_answer": "answer",
        })
    summary = {
        "schema": runner.RUN_SCHEMA,
        "status": "S38_ALL_LOCAL_BASELINE_COMPLETE",
        "host_label": host,
        "requests": 2,
        "arrival_mode": "trace",
        "workers": 2,
        "retrieval_top_k": 1,
        "rerank_top_n": 1,
        "requests_per_s": 1.0,
        "realized_output_tokens_per_s": 3.0,
        "requested_input_tokens": 20,
        "requested_output_tokens": 8,
        "realized_prompt_tokens": 24,
        "realized_output_tokens": 6,
        "client_queue_ms": {"p50": 0.005, "p95": 0.005, "max": 0.005},
        "service_ms": {"p50": 0.02, "p95": 0.02, "max": 0.02},
        "response_ms": {
            "p50": 0.025,
            "p95": 0.026 if mutate else 0.025,
            "max": 0.026 if mutate else 0.025,
        },
        "models": {"embedding_sha256": "a" * 64},
        "inputs": {"trace_sha256": "b" * 64},
    }
    directory.mkdir()
    requests_path = directory / "requests.jsonl"
    summary_path = directory / "summary.json"
    requests_path.write_text("".join(runner.canonical_json(row) + "\n" for row in rows), encoding="ascii")
    summary_path.write_text(runner.canonical_json(summary) + "\n", encoding="ascii")
    manifest = {
        "schema": runner.RUN_SCHEMA,
        "requests": {"path": "requests.jsonl", "sha256": compare_tool.file_sha256(requests_path)},
        "summary": {"path": "summary.json", "sha256": compare_tool.file_sha256(summary_path)},
    }
    (directory / "manifest.json").write_text(runner.canonical_json(manifest) + "\n", encoding="ascii")


class ServerBaselineTests(unittest.TestCase):
    def test_answer_scores(self) -> None:
        self.assertEqual(runner.answer_scores("The Eiffel Tower", "the eiffel tower"), (1.0, 1.0))
        exact, f1 = runner.answer_scores("Eiffel", "Eiffel Tower")
        self.assertEqual(exact, 0.0)
        self.assertAlmostEqual(f1, 2 / 3)

    def test_retrieve_deduplicates_documents(self) -> None:
        chunks = [
            {"chunk_id": "a0", "document_id": "a", "text": "", "title": "", "url": "", "shard": "OP12"},
            {"chunk_id": "a1", "document_id": "a", "text": "", "title": "", "url": "", "shard": "OP12"},
            {"chunk_id": "b0", "document_id": "b", "text": "", "title": "", "url": "", "shard": "OP15"},
        ]
        matrix = np.asarray([[1.0, 0.0], [0.9, 0.0], [0.8, 0.0]], dtype=np.float32)
        result = runner.retrieve(np.asarray([1.0, 0.0], dtype=np.float32), chunks, matrix, 2)
        self.assertEqual([item["chunk_id"] for item in result], ["a0", "b0"])

    def test_cohort_is_deterministic_and_rebased(self) -> None:
        requests = []
        for index in range(20):
            requests.append({
                "event_id": f"event-{index}",
                "t_us": 100 + index,
                "input_tokens": 100 + index,
                "output_tokens": 10 + index,
                "source_fields": {
                    "question_type": "comparison" if index % 2 else "temporal",
                    "evidence_count": 2,
                },
            })
        first = selector.select(requests, 8)
        second = selector.select(list(reversed(requests)), 8)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["t_us"], 0)
        self.assertEqual(len({item["event_id"] for item in first}), 8)

    def test_endpoint_model_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            path.write_bytes(b"model")
            digest = hashlib.sha256(b"model").hexdigest()
            runner.verify_endpoint_model("test", {"model_path": str(path)}, digest)
            with self.assertRaises(runner.RunError):
                runner.verify_endpoint_model("test", {"model_path": str(path)}, "0" * 64)

    def test_matched_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_result(root / "left", "left")
            write_result(root / "right", "right")
            left_summary, left_rows = compare_tool.validate_result(root / "left")
            right_summary, right_rows = compare_tool.validate_result(root / "right")
            result = compare_tool.compare(left_summary, left_rows, right_summary, right_rows)
            self.assertEqual(result["status"], "S38_MATCHED_SERVER_BASELINES_VALID")
            self.assertEqual(result["relative"]["generated_answer_exact_agreement"], 1.0)

    def test_timeline_mutation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad"
            write_result(path, "bad", mutate=True)
            with self.assertRaisesRegex(compare_tool.ComparisonError, "response mismatch"):
                compare_tool.validate_result(path)


if __name__ == "__main__":
    unittest.main()
