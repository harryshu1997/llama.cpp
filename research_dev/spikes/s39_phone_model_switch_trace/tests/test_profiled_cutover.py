#!/usr/bin/env python3
"""Focused fail-closed tests for the W9 profiled cutover gate."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent.parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
for path in (str(HERE), str(S22)):
    if path not in sys.path:
        sys.path.insert(0, path)

import build_w9_contract as builder
import cuda_profiled_trace_control_probe as control
import phone_cuda_delta_probe as w6
import phone_cuda_profiled_cutover_probe as treatment
import run_w9_profiled_cutover_gate as campaign
import validate_profiled_cutover_pair as pair
import validate_profiled_cutover_series as series
import w9_host_evidence as evidence
import w9_profiled_cutover as w9


class FakeLedger:
    def __init__(self) -> None:
        self.items = []

    def append(self, event, payload, event_ns=None):
        value = {
            "event": event,
            "event_ns": event_ns or 100 + len(self.items),
            "payload": payload,
        }
        record = w9.LedgerRecord(
            f"{len(self.items):06d}.json",
            "a" * 64,
            value,
        )
        self.items.append(record)
        return record, value["event_ns"]


class W9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.contract_path = Path(cls.temporary.name) / "contract.json"
        w9.write_atomic(cls.contract_path, builder.build())
        cls.contract = w9.load_contract(cls.contract_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_current_profile_selects_zero(self):
        decision = w9.select_cutover(
            self.contract,
            phone_tokens_at_f0=2,
            inflight_present=True,
            inflight_elapsed_us=585480,
        )
        self.assertEqual(decision.k_extra, 0)
        self.assertEqual(decision.k_max, 9)

    def test_no_inflight_selects_zero(self):
        decision = w9.select_cutover(
            self.contract,
            phone_tokens_at_f0=2,
            inflight_present=False,
            inflight_elapsed_us=0,
        )
        self.assertEqual(decision.k_extra, 0)
        self.assertEqual(decision.predicted_inflight_remaining_us, 0)

    def test_absent_inflight_rejects_elapsed(self):
        with self.assertRaisesRegex(w9.W9Error, "absent batch"):
            w9.select_cutover(
                self.contract,
                phone_tokens_at_f0=2,
                inflight_present=False,
                inflight_elapsed_us=1,
            )

    def test_kmax_zero(self):
        decision = w9.select_cutover(
            self.contract,
            phone_tokens_at_f0=11,
            inflight_present=True,
            inflight_elapsed_us=0,
        )
        self.assertEqual(decision.k_max, 0)
        self.assertEqual(decision.k_extra, 0)

    def test_faster_phone_selects_positive(self):
        fast = replace(
            self.contract,
            cuda_replay_us=5_000_000,
            phone_batch_estimate_us=100_000,
            phone_extra_us=tuple(150_000 * i for i in range(12)),
        )
        decision = w9.select_cutover(
            fast,
            phone_tokens_at_f0=2,
            inflight_present=True,
            inflight_elapsed_us=0,
        )
        self.assertGreater(decision.k_extra, 0)

    def test_margin_boundary_is_inclusive(self):
        boundary = replace(
            self.contract,
            cuda_replay_us=200_000,
            phone_batch_estimate_us=50_000,
            phone_extra_us=tuple(150_000 * i for i in range(12)),
            delta_ingest_us=(0,) * 13,
            cuda_tokens_us=tuple(100_000 * i for i in range(14)),
        )
        decision = w9.select_cutover(
            boundary,
            phone_tokens_at_f0=2,
            inflight_present=True,
            inflight_elapsed_us=50_000,
        )
        self.assertTrue(decision.candidates[1]["feasible"])
        slower = replace(
            boundary,
            phone_extra_us=(0,) + tuple(
                150_001 + 150_000 * (i - 1) for i in range(1, 12)
            ),
        )
        decision = w9.select_cutover(
            slower,
            phone_tokens_at_f0=2,
            inflight_present=True,
            inflight_elapsed_us=50_000,
        )
        self.assertFalse(decision.candidates[1]["feasible"])

    def test_flat_predictors_are_deterministic(self):
        flat = replace(
            self.contract,
            cuda_replay_us=1_000_000,
            phone_batch_estimate_us=100_000,
            phone_extra_us=(0,) * 12,
            delta_ingest_us=(0,) * 13,
            cuda_tokens_us=(0,) * 14,
        )
        decision = w9.select_cutover(
            flat,
            phone_tokens_at_f0=2,
            inflight_present=True,
            inflight_elapsed_us=0,
        )
        self.assertEqual(decision.k_extra, decision.k_max)

    def test_no_feasible_positive_keeps_zero(self):
        slow = replace(
            self.contract,
            cuda_replay_us=100_000,
            phone_batch_estimate_us=1_000_000,
            phone_extra_us=tuple(1_000_000 * i for i in range(12)),
        )
        decision = w9.select_cutover(
            slow,
            phone_tokens_at_f0=2,
            inflight_present=True,
            inflight_elapsed_us=0,
        )
        self.assertEqual(decision.k_extra, 0)
        self.assertTrue(all(
            not candidate["feasible"]
            for candidate in decision.candidates[1:]
        ))

    def test_deterministic_decision(self):
        values = [
            w6.canonical(
                w9.select_cutover(
                    self.contract,
                    phone_tokens_at_f0=3,
                    inflight_present=True,
                    inflight_elapsed_us=700000,
                ).as_dict()
            )
            for _ in range(20)
        ]
        self.assertEqual(len(set(values)), 1)

    def test_contract_rejects_float(self):
        value = builder.build()
        value["policy"]["cuda_replay_us"] = 180000.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            w9.write_atomic(path, value)
            with self.assertRaisesRegex(w9.W9Error, "nonnegative integer"):
                w9.load_contract(path)

    def test_contract_rejects_nonmonotone_predictor(self):
        value = builder.build()
        value["policy"]["phone_extra_us"][2] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            w9.write_atomic(path, value)
            with self.assertRaisesRegex(w9.W9Error, "not monotone"):
                w9.load_contract(path)

    def test_contract_rejects_source_mutation(self):
        contract = replace(
            self.contract,
            source_sha256={
                **self.contract.source_sha256,
                "w9_profiled_cutover.py": "f" * 64,
            },
        )
        original = w9.load_contract
        with mock.patch.object(w9, "load_contract", return_value=contract):
            with self.assertRaisesRegex(w9.W9Error, "source mismatch"):
                pair.load_dependencies(
                    self.contract_path,
                    HERE / "W8_LIVE_SESSION_CONTRACT_R1.json",
                    HERE / "W5_HANDOFF_CONTRACT.json",
                    HERE / "W6_DELTA_CONTRACT.json",
                    HERE / "W6_PHYSICAL_GATE.json",
                )
        self.assertIs(w9.load_contract, original)

    def test_w8_profile_source_is_reopened_and_bounded(self):
        pair.validate_w8_profile_source(self.contract)
        mutated = replace(
            self.contract,
            w8_treatment_report_sha256="f" * 64,
        )
        with self.assertRaisesRegex(w9.W9Error, "source identity"):
            pair.validate_w8_profile_source(mutated)

    def test_ledger_round_trip_and_hash_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger"
            ledger = w9.PublicationLedger(
                path,
                transaction_id="a" * 64,
                run_id="b" * 64,
                request_ids=[1, 2],
            )
            ledger.append("PAID_START", {"owner": "PHONE"}, event_ns=10)
            ledger.append("COMPLETE", {"owner": "NONE"}, event_ns=20)
            records = w9.load_ledger(
                path,
                transaction_id="a" * 64,
                run_id="b" * 64,
                request_ids=[1, 2],
            )
            self.assertEqual([r.value["event"] for r in records], [
                "PAID_START",
                "COMPLETE",
            ])

    def test_ledger_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger"
            ledger = w9.PublicationLedger(
                path,
                transaction_id="a" * 64,
                run_id="b" * 64,
                request_ids=[1],
            )
            ledger.append("PAID_START", {}, event_ns=10)
            ledger.append("COMPLETE", {}, event_ns=20)
            record = path / "000000.json"
            value = json.loads(record.read_text(encoding="ascii"))
            value["payload"]["tamper"] = True
            record.write_bytes(w6.canonical(value))
            with self.assertRaisesRegex(w9.W9Error, "invalid record"):
                w9.load_ledger(
                    path,
                    transaction_id="a" * 64,
                    run_id="b" * 64,
                    request_ids=[1],
                )

    def test_ledger_rejects_stale_epoch_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger"
            ledger = w9.PublicationLedger(
                path,
                transaction_id="a" * 64,
                run_id="b" * 64,
                request_ids=[1],
            )
            ledger.append("PAID_START", {}, event_ns=10)
            ledger.append("COMPLETE", {}, event_ns=20)
            second = path / "000001.json"
            value = json.loads(second.read_text(encoding="ascii"))
            value["previous_record_sha256"] = "f" * 64
            second.write_bytes(w6.canonical(value))
            with self.assertRaisesRegex(w9.W9Error, "invalid record"):
                w9.load_ledger(
                    path,
                    transaction_id="a" * 64,
                    run_id="b" * 64,
                    request_ids=[1],
                )

    def test_publication_round_is_rectangular(self):
        ledger = FakeLedger()
        times = treatment.publish_round(
            ledger,
            owner="PHONE",
            owner_epoch=1,
            request_ids=[10, 11],
            tokens=[20, 21],
            positions=[8, 8],
            classification="PHONE_F0",
        )
        self.assertEqual(len(times), 2)
        self.assertEqual(len(ledger.items), 2)

    def test_publication_round_rejects_partial_batch(self):
        with self.assertRaisesRegex(w9.W9Error, "invalid token round"):
            treatment.publish_round(
                FakeLedger(),
                owner="PHONE",
                owner_epoch=1,
                request_ids=[10, 11],
                tokens=[20],
                positions=[8],
                classification="PHONE_PRE_READY",
            )

    def test_control_trace_zero_delta(self):
        sequences = []
        for index in range(self.contract.batch):
            sequences.append({
                "cuda_continuation": list(range(12)),
                "inflight_phone_tokens": [],
                "phone_service": [1],
                "preexisting_tokens": [2, 3],
                "prompt_tokens": list(range(8)),
                "sequence_index": index,
            })
        value = {
            "batch": 8,
            "contract_sha256": self.contract.raw_sha256,
            "pair_ordinal": "P1",
            "run_id": "a" * 64,
            "schema": "s39-profiled-cutover-treatment-v1",
            "sequences": sequences,
            "status": "PROFILED_ZERO_EXTRA_TREATMENT_PASS",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            w9.write_atomic(path, value)
            histories, trace, _ = control.load_trace(
                path,
                self.contract,
                "a" * 64,
                "P1",
            )
            self.assertEqual(len(histories[0]), 10)
            self.assertEqual(len(trace[0]), 13)

    def test_control_trace_rejects_budget_error(self):
        value = {
            "batch": 8,
            "contract_sha256": self.contract.raw_sha256,
            "pair_ordinal": "P1",
            "run_id": "a" * 64,
            "schema": "s39-profiled-cutover-treatment-v1",
            "sequences": [
                {
                    "cuda_continuation": [],
                    "inflight_phone_tokens": [],
                    "phone_service": [],
                    "preexisting_tokens": [1, 2],
                    "prompt_tokens": list(range(8)),
                    "sequence_index": index,
                }
                for index in range(8)
            ],
            "status": "PROFILED_ZERO_EXTRA_TREATMENT_PASS",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            w9.write_atomic(path, value)
            with self.assertRaisesRegex(w9.W9Error, "treatment tokens"):
                control.load_trace(path, self.contract, "a" * 64, "P1")

    def test_even_median_uses_middle_mean(self):
        from fractions import Fraction

        self.assertEqual(
            series.even_median([
                Fraction(1, 4),
                Fraction(3, 4),
                Fraction(1, 2),
                Fraction(1, 1),
            ]),
            Fraction(5, 8),
        )

    def test_even_median_rejects_wrong_count(self):
        from fractions import Fraction

        with self.assertRaisesRegex(w9.W9Error, "four"):
            series.even_median([Fraction(1, 2)])

    def test_gpu_parser_rejects_wrong_shape(self):
        with self.assertRaisesRegex(w9.W9Error, "shape"):
            evidence.parse_csv_line(b"one,two\n", 3, "gpu")

    def test_gpu_decimal_is_integer_milli(self):
        self.assertEqual(evidence.decimal_milli("300.00", "power"), 300000)

    def test_gpu_float_integer_rejected(self):
        with self.assertRaisesRegex(w9.W9Error, "invalid integer"):
            evidence.integer("1.0", "gpu")

    def test_raw_gpu_evidence_round_trip(self):
        raw = (
            "0, NVIDIA RTX A6000, GPU-" + "a" * 32
            + ", 00000000:51:00.0, 580.159.03, 49140, 300.00, "
            "41, P8, 210, 405, 25.03, 108, 0\n"
        )
        parsed = pair.parse_raw_gpu(raw, "gpu")
        self.assertEqual(parsed["index"], 0)
        self.assertEqual(parsed["power_limit_mw"], 300000)
        self.assertEqual(parsed["utilization_gpu_pct"], 0)

    def test_raw_process_evidence_round_trip(self):
        uuid = "GPU-" + "a" * 32
        parsed = pair.parse_raw_processes(
            f"{uuid}, 123, llama-layersplit, 4096\n",
            "process",
        )
        self.assertEqual(parsed, [{
            "gpu_uuid": uuid,
            "pid": 123,
            "process_name": "llama-layersplit",
            "used_memory_mib": 4096,
        }])

    def test_launch_exact_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launch.json"
            w9.write_atomic(path, {
                "cuda_launch_ns": 1_100_000_000,
                "launch_deadline_ns": 1_100_000_000,
                "pair_ordinal": "P1",
                "phase": "TREATMENT",
                "preexisting_route_pids": [],
                "request_start_ns": 1_000_000_000,
                "run_id": "a" * 64,
                "schema": "s39-profiled-cutover-launch-v1",
            })
            value, _ = pair.validate_launch(
                path,
                phase="TREATMENT",
                pair="P1",
                run_id="a" * 64,
                contract=self.contract,
            )
            self.assertEqual(value["cuda_launch_ns"], 1_100_000_000)

    def test_launch_rejects_preexisting_process(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launch.json"
            w9.write_atomic(path, {
                "cuda_launch_ns": 1_100_000_000,
                "launch_deadline_ns": 1_100_000_000,
                "pair_ordinal": "P1",
                "phase": "TREATMENT",
                "preexisting_route_pids": [1],
                "request_start_ns": 1_000_000_000,
                "run_id": "a" * 64,
                "schema": "s39-profiled-cutover-launch-v1",
            })
            with self.assertRaisesRegex(w9.W9Error, "launch timing"):
                pair.validate_launch(
                    path,
                    phase="TREATMENT",
                    pair="P1",
                    run_id="a" * 64,
                    contract=self.contract,
                )

    def test_manifest_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            path = root / "value.txt"
            path.write_text("one", encoding="ascii")
            owner = object.__new__(campaign.Campaign)
            owner.hash_tree(root)
            owner.verify_manifest(root)
            path.write_text("two", encoding="ascii")
            with self.assertRaisesRegex(w9.W9Error, "manifest mismatch"):
                owner.verify_manifest(root)

    def test_manifest_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            (root / "value.txt").write_text("one", encoding="ascii")
            owner = object.__new__(campaign.Campaign)
            owner.hash_tree(root)
            with self.assertRaisesRegex(w9.W9Error, "already exists"):
                owner.hash_tree(root)

    def test_atomic_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            w9.write_atomic(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                w9.write_atomic(path, {"a": 2})


if __name__ == "__main__":
    unittest.main()
