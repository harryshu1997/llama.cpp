#!/usr/bin/env python3
"""Focused offline tests for the replay-partition diagnostic."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_contract as builder
import replay_partition as rp
import validate_replay_partition_diagnostic as validator
from stage_v3_client import BatchResult, Status


class FakeClient:
    def __init__(self) -> None:
        self.active: set[int] = set()

    def batch(self, rows):
        for row in rows:
            self.active.add(row.seq_id)
        return tuple(
            BatchResult(
                row.request_id,
                row.route_epoch,
                row.seq_id,
                row.position,
                None,
                (row.token + row.position + len(rows)) % 100003,
            )
            for row in rows
        )

    def status(self):
        return Status(len(self.active), 8, False)

    def remove(self, seq_id, request_id, route_epoch):
        if request_id != route_epoch or seq_id not in self.active:
            raise RuntimeError("invalid fake removal")
        self.active.remove(seq_id)
        return self.status()


class ReplayPartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = builder.build_inputs()
        cls.contract = builder.build_contract("a" * 64)

    def test_exact_w9_histories(self):
        self.assertEqual(self.inputs["f0_history_sha256"], builder.F0_SHA256)
        self.assertEqual(self.inputs["f1_history_sha256"], builder.F1_SHA256)
        self.assertEqual(
            self.inputs["w9_continuation_sha256"],
            builder.W9_CONTINUATION_SHA256,
        )
        self.assertEqual([len(row) for row in self.inputs["f0_histories"]], [11] * 8)
        self.assertEqual([len(row) for row in self.inputs["f1_histories"]], [12] * 8)

    def test_contract_run_order(self):
        self.assertEqual(
            [run["name"] for run in self.contract["execution"]["runs"]],
            [
                "fresh_incremental_r1",
                "fresh_incremental_r2",
                "fresh_full_chunk2_r1",
                "fresh_full_chunk2_r2",
                "same_process_incremental_then_full",
                "fresh_full_8_4_r1",
                "fresh_full_8_4_r2",
            ],
        )

    def test_incremental_records_8_3_plus_1_before_continuation(self):
        client = rp.RecordingClient(FakeClient())
        spec = builder.path("incremental_8_3_plus_1", 10000)
        result = rp.execute_path(client, spec, self.inputs, 11)
        replay = [call for call in client.calls if call["phase"] == "REPLAY_F0"]
        delta = [
            call
            for call in client.calls
            if call["phase"] == "INGEST_F1_MINUS_F0"
        ]
        continuation = [
            call
            for call in client.calls
            if call["phase"] == "AUTONOMOUS_CONTINUATION"
        ]
        self.assertEqual([call["shape"]["rows_per_sequence"][0] for call in replay], [8, 3])
        self.assertEqual([call["shape"]["rows_per_sequence"][0] for call in delta], [1])
        self.assertEqual(len(continuation), 10)
        self.assertEqual(result["state_counts"], {
            "after_remove": 0,
            "before": 0,
            "before_remove": 8,
        })

    def test_full_chunk2_records_six_calls(self):
        client = rp.RecordingClient(FakeClient())
        spec = builder.path("full_f1_chunk2", 20000)
        result = rp.execute_path(client, spec, self.inputs, 11)
        replay = [call for call in client.calls if call["phase"] == "REPLAY_F1"]
        self.assertEqual(len(replay), 6)
        self.assertEqual(
            [call["shape"]["rows_per_sequence"][0] for call in replay],
            [2] * 6,
        )
        self.assertEqual(len(result["continuation"][0]), 11)

    def test_same_process_state_is_empty_between_paths(self):
        client = rp.RecordingClient(FakeClient())
        first = rp.execute_path(
            client,
            builder.path("incremental_8_3_plus_1", 30000),
            self.inputs,
            11,
        )
        second = rp.execute_path(
            client,
            builder.path("full_f1_chunk2", 40000),
            self.inputs,
            11,
        )
        self.assertEqual(first["state_counts"]["after_remove"], 0)
        self.assertEqual(second["state_counts"]["before"], 0)
        self.assertEqual(client.status().active_sequences, 0)

    def test_validator_reconstructs_recorded_continuation(self):
        client = rp.RecordingClient(FakeClient())
        spec = builder.path("full_f1_8_4", 50000)
        path = rp.execute_path(client, spec, self.inputs, 11)
        report = {"calls": client.calls}
        reconstructed = validator.validate_path(
            report,
            path,
            spec,
            self.inputs,
            self.contract,
        )
        self.assertEqual(reconstructed, path["continuation"])

    def test_first_mismatch_is_positioned(self):
        left = [[1, 2], [3, 4]]
        right = [[1, 2], [3, 5]]
        self.assertEqual(
            rp.first_mismatch(left, right),
            {
                "absolute_position": 13,
                "left_token": 4,
                "right_token": 5,
                "sequence_index": 1,
                "token_offset": 1,
            },
        )

    def test_duplicate_json_key_rejected(self):
        with self.assertRaisesRegex(rp.DiagnosticError, "duplicate JSON key"):
            json.loads('{"x":1,"x":2}', object_pairs_hook=rp.strict_object)


if __name__ == "__main__":
    unittest.main()
