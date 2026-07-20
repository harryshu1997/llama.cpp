#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
S15 = HERE.parent
S14 = S15.parent / "s14_mixed_streaming_scheduler"
sys.path.insert(0, str(S15))
sys.path.insert(0, str(S14))

import trace_adapter as T  # noqa: E402
from runtime_dispatch import TERMINAL_STATES  # noqa: E402


class TraceIngestTests(unittest.TestCase):
    def test_trace_loads_in_deterministic_order(self) -> None:
        rows = T.load_trace()
        self.assertEqual(len(rows), 177)
        arrivals = [(row["t_us"], row["event_id"]) for row in rows]
        self.assertEqual(arrivals, sorted(arrivals))

    def test_trace_deadline_and_priority_stay_null(self) -> None:
        for row in T.load_trace():
            self.assertIsNone(row["deadline_us"])
            self.assertIsNone(row["priority_class"])

    def test_rejects_a_row_that_already_carries_a_deadline(self) -> None:
        with self.assertRaisesRegex(T.TraceAdapterError, "duplicate key|already carries"):
            path = HERE / "_scratch_bad_trace.jsonl"
            path.write_text(
                '{"t_us":1,"event_id":"x","service":"api_generation","source":"s",'
                '"deadline_us":5,"priority_class":1}\n', encoding="ascii")
            try:
                T.load_trace(path)
            finally:
                path.unlink()


class SidecarTests(unittest.TestCase):
    def test_sidecar_is_explicitly_synthetic(self) -> None:
        self.assertEqual(T.SIDECAR["provenance"], "s15-synthetic-sidecar")

    def test_sidecar_assigns_priority_and_relative_deadline(self) -> None:
        rows = T.load_trace()
        planned = T.apply_sidecar(rows)
        self.assertEqual(len(planned), len(rows))
        for request, row in zip(planned, rows):
            self.assertIn(request.priority_class, (0, 1))
            budget = T.SIDECAR["deadline_budget_us_by_priority"][request.priority_class]
            self.assertEqual(request.deadline_us, row["t_us"] + budget)
            self.assertIn(request.island_id, T.SIDECAR["phone_islands"])

    def test_islands_round_robin_is_deterministic(self) -> None:
        planned = T.apply_sidecar(T.load_trace())
        islands = [request.island_id for request in planned]
        self.assertEqual(islands[0], "gemma-head-0-8")
        self.assertEqual(islands[1], "gemma-head-0-6")


class ReplayTests(unittest.TestCase):
    def test_replay_conserves_every_request(self) -> None:
        decisions = T.replay()
        self.assertEqual(len(decisions), 177)
        ids = [row["event_id"] for row in decisions]
        self.assertEqual(len(set(ids)), 177)
        for row in decisions:
            self.assertIn(row["terminal_state"], TERMINAL_STATES)

    def test_replay_terminal_counts_match_priority_policy(self) -> None:
        decisions = T.replay()
        counts = {}
        for row in decisions:
            counts[row["terminal_state"]] = counts.get(row["terminal_state"], 0) + 1
        # 152 low-priority api_generation to phones, 25 interactive to server.
        self.assertEqual(counts["completed_phone"], 152)
        self.assertEqual(counts["completed_server"], 25)
        self.assertNotIn("timed_out", counts)

    def test_summary_verdict_wording(self) -> None:
        summary = T.summarize(T.replay())
        self.assertEqual(
            summary["verdict"],
            "RUNTIME_DISPATCH_MECHANICS_PASS_PHYSICAL_EXECUTION_NOT_RUN",
        )


class DeterminismTests(unittest.TestCase):
    def test_in_process_log_is_stable(self) -> None:
        first = T.emit_log(T.replay())
        second = T.emit_log(T.replay())
        self.assertEqual(first, second)

    def test_log_is_byte_identical_across_processes_and_hash_seeds(self) -> None:
        digests = set()
        for seed in ("0", "1", "12345"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            proc = subprocess.run(
                [sys.executable, str(S15 / "trace_adapter.py")],
                capture_output=True, check=True, env=env,
            )
            digests.add(hashlib.sha256(proc.stdout).hexdigest())
        self.assertEqual(len(digests), 1)


if __name__ == "__main__":
    unittest.main()
