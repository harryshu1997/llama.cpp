#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from physical_adapter import load_trace, summarize_run
from priority_runtime import (
    PhysicalRuntimeError,
    prepare_trace,
    validate_conservation,
)


TRACE = HERE.parent / "s24_overlap_handoff_poc/deterministic-three-class.json"


def completed(request_id: int, route_id: str, epoch: int, priority: int) -> dict:
    slo_us = {0: 2_000_000, 1: 8_000_000, 2: 15_000_000}[priority]
    return {
        "request_id": request_id,
        "route_id": route_id,
        "route_epoch": epoch,
        "priority": priority,
        "prompt_length": 1,
        "output_steps": 4,
        "scheduled_arrival_ns": 0,
        "slo_met": True,
        "latency_us": 100,
        "slo_us": slo_us,
        "ttft_us": 50,
        "admission_queue_us": 1,
        "lease_queue_us": 1,
    }


def valid_records() -> tuple[list, list[dict], list[dict], list[dict]]:
    rows = prepare_trace(load_trace(TRACE))
    decisions = []
    results = []
    epoch = 1
    for request_id in range(24001, 24005):
        decisions.append({
            "route_id": "R0",
            "request_ids": [request_id],
            "route_epochs": [epoch],
            "priorities": [0],
            "batch_size": 1,
            "predicted_cuda_relief_us": 0,
        })
        results.append(completed(request_id, "R0", epoch, 0))
        epoch += 1
    for request_ids, priority in (
        (list(range(24005, 24009)), 1),
        (list(range(24009, 24013)), 2),
    ):
        epochs = list(range(epoch, epoch + 4))
        decisions.append({
            "route_id": "R2",
            "request_ids": request_ids,
            "route_epochs": epochs,
            "priorities": [priority] * 4,
            "batch_size": 4,
            "predicted_cuda_relief_us": 420779,
        })
        results.extend(
            completed(request_id, "R2", route_epoch, priority)
            for request_id, route_epoch in zip(request_ids, epochs)
        )
        epoch += 4
    return rows, decisions, results, []


class ConservationTests(unittest.TestCase):
    def test_frozen_trace_is_prepared_in_stable_order(self) -> None:
        rows = prepare_trace(load_trace(TRACE))
        self.assertEqual(len(rows), 12)
        self.assertEqual([row.request_id for row in rows], list(range(24001, 24013)))

    def test_valid_priority_ownership(self) -> None:
        validate_conservation(*valid_records())

    def test_explicit_measured_batch_set_is_accepted(self) -> None:
        validate_conservation(
            *valid_records(),
            {"R0": frozenset((1, 4, 32)), "R2": frozenset((1, 4, 32))},
        )

    def test_invalid_measured_batch_set_is_rejected(self) -> None:
        with self.assertRaisesRegex(PhysicalRuntimeError, "batch points"):
            validate_conservation(
                *valid_records(),
                {"R0": frozenset((1, 4)), "R2": frozenset((1, "bad"))},
            )

    def test_duplicate_dispatch_is_rejected(self) -> None:
        rows, decisions, results, rejected = valid_records()
        decisions.append(dict(decisions[0]))
        with self.assertRaisesRegex(PhysicalRuntimeError, "multiple dispatch"):
            validate_conservation(rows, decisions, results, rejected)

    def test_urgent_offload_is_rejected(self) -> None:
        rows, decisions, results, rejected = valid_records()
        decisions[0] = {
            **decisions[0],
            "route_id": "R2",
            "predicted_cuda_relief_us": 1,
        }
        results[0] = {**results[0], "route_id": "R2"}
        with self.assertRaisesRegex(PhysicalRuntimeError, "urgent work"):
            validate_conservation(rows, decisions, results, rejected)

    def test_unmeasured_batch_is_rejected(self) -> None:
        rows, decisions, results, rejected = valid_records()
        decisions[4]["batch_size"] = 3
        decisions[4]["request_ids"] = decisions[4]["request_ids"][:3]
        decisions[4]["route_epochs"] = decisions[4]["route_epochs"][:3]
        decisions[4]["priorities"] = decisions[4]["priorities"][:3]
        with self.assertRaises(PhysicalRuntimeError):
            validate_conservation(rows, decisions, results, rejected)

    def test_completion_lineage_is_rejected(self) -> None:
        rows, decisions, results, rejected = valid_records()
        results[0] = {**results[0], "route_epoch": 999}
        with self.assertRaisesRegex(PhysicalRuntimeError, "lineage"):
            validate_conservation(rows, decisions, results, rejected)

    def test_trace_priority_mutation_is_rejected(self) -> None:
        rows, decisions, results, rejected = valid_records()
        decisions[0]["priorities"] = [1]
        results[0] = {**results[0], "priority": 1}
        with self.assertRaisesRegex(PhysicalRuntimeError, "differs from the trace"):
            validate_conservation(rows, decisions, results, rejected)

    def test_slo_boolean_is_recomputed(self) -> None:
        rows, decisions, results, rejected = valid_records()
        results[0] = {**results[0], "latency_us": results[0]["slo_us"] + 1}
        with self.assertRaisesRegex(PhysicalRuntimeError, "SLO result"):
            validate_conservation(rows, decisions, results, rejected)


class SummaryTests(unittest.TestCase):
    def test_summary_recomputes_priority_and_cuda_work(self) -> None:
        _rows, _decisions, results, rejected = valid_records()
        runtime = {
            "requests": results,
            "rejected": rejected,
            "duration_ns": 2_000_000,
        }
        events = {
            name: []
            for name in (
                "cuda-prefix", "cuda-mid", "op12-prefix", "op15-mid", "cuda-tail",
            )
        }
        events["cuda-tail"] = [{
            "batch_size": 4,
            "compute_us": 10,
            "dispatch_ns": 1,
            "dispatch_reason": "BATCH_KNEE",
            "status": "OK",
        }]
        summary = summarize_run(runtime, events)
        self.assertEqual(summary["completed_requests"], 12)
        self.assertEqual(summary["priority"]["0"]["completed"], 4)
        self.assertEqual(summary["summed_cuda_island_compute_us"], 10)


if __name__ == "__main__":
    unittest.main()
