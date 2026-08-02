#!/usr/bin/env python3

import pathlib
import sys
import tempfile
import time
import unittest
from dataclasses import asdict
from types import SimpleNamespace


HERE = pathlib.Path(__file__).parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cuda_live_control_probe as control
import cuda_live_trace_control_probe as trace_control
import phone_cuda_cold_promotion_probe as cold
import phone_cuda_live_promotion_probe as live
import validate_live_promotion as validator
import validate_live_promotion_r1 as validator_r1


class FakeClient:
    def __init__(self):
        self.histories = {}

    @staticmethod
    def predict(history):
        return (
            sum((index + 1) * token for index, token in enumerate(history))
            + 31
        ) % 50000

    def batch(self, rows):
        results = []
        for row in rows:
            history = self.histories.setdefault(row.seq_id, [])
            if row.position != len(history):
                raise live.ProtocolError("noncontiguous fake history")
            history.append(row.token)
            results.append(live.w6.BatchResult(
                row.request_id,
                row.route_epoch,
                row.seq_id,
                row.position,
                None,
                self.predict(history),
            ))
        return tuple(results)

    def remove(self, seq_id, request_id, route_epoch):
        del request_id, route_epoch
        del self.histories[seq_id]
        return SimpleNamespace(
            active_sequences=len(self.histories),
            draining=False,
            max_streams=64,
        )


class BoundaryReady:
    def __init__(self, ready_check):
        self.ready_check = ready_check
        self.checks = 0
        self.ready_ns = None

    @staticmethod
    def raise_if_failed():
        return None

    def is_ready(self):
        self.checks += 1
        if self.checks >= self.ready_check and self.ready_ns is None:
            self.ready_ns = time.monotonic_ns()
        return self.ready_ns is not None


class LivePromotionTests(unittest.TestCase):
    RUN_ID = "b" * 64

    @classmethod
    def setUpClass(cls):
        cls.base = live.w5.load_contract(cold.DEFAULT_BASE_CONTRACT)
        cls.delta = live.w6.load_contract(
            cold.DEFAULT_DELTA_CONTRACT,
            cls.base,
        )
        cls.gate = validator.physical.load_physical_gate(
            cold.DEFAULT_PHYSICAL_GATE,
            cls.delta,
            cls.base,
        )
        cls.contract = live.load_contract(
            live.DEFAULT_CONTRACT,
            cls.base,
            cls.delta,
            cls.gate.raw_sha256,
        )
        cls.r1_contract = live.load_contract(
            HERE / "W8_LIVE_SESSION_CONTRACT_R1.json",
            cls.base,
            cls.delta,
            cls.gate.raw_sha256,
        )

    def prompts(self):
        return [
            [
                seq_id * 100 + position
                for position in range(self.base.prompt_tokens)
            ]
            for seq_id in range(self.contract.batch)
        ]

    def hello(self):
        return {
            "capabilities": 0x3F,
            "file_type": self.base.file_type,
            "layer_end": self.base.n_layer,
            "layer_start": 0,
            "max_streams": self.contract.batch,
            "model_sha256": self.base.model_sha256,
            "n_batch": self.contract.max_rows_per_batch,
            "n_ctx_seq": 256,
            "n_embd": self.base.n_embd,
            "n_layer": self.base.n_layer,
            "n_ubatch": self.contract.max_rows_per_batch,
        }

    def treatment_flow(self, journal_path):
        prompts = self.prompts()
        phone = FakeClient()
        cuda = FakeClient()
        preexisting, pre_metrics = cold.run_fixed_generation(
            phone,
            prompts,
            10000,
            self.contract.phone_prefill_chunk,
            self.contract.preexisting_committed_tokens,
        )
        connector = BoundaryReady(3)
        service, service_metrics = live.continue_active_until_ready(
            phone,
            [tokens[-1] for tokens in preexisting],
            10000,
            self.base.prompt_tokens
            + self.contract.preexisting_committed_tokens
            - 1,
            self.contract.min_phone_service_tokens,
            self.contract.max_phone_service_tokens,
            connector,
        )
        service_count = len(service[0])
        histories = [
            prompt + before + during
            for prompt, before, during in zip(prompts, preexisting, service)
        ]
        phone_result, cuda_result, concurrency = live.w6.run_concurrently(
            lambda: live.w6.advance_active(
                phone,
                [tokens[-1] for tokens in service],
                10000,
                self.base.prompt_tokens
                + self.contract.preexisting_committed_tokens
                + service_count
                - 1,
                self.contract.phone_delta_tokens,
            ),
            lambda: live.w6.replay_history_only(
                cuda,
                histories,
                30000,
                self.contract.cuda_replay_chunk,
            ),
            10,
        )
        concurrency = {
            "phone_started_ns": 100,
            "phone_ended_ns": 1100,
            "phone_wall_ns": 1000,
            "cuda_started_ns": 100,
            "cuda_ended_ns": 900,
            "cuda_wall_ns": 800,
            "overlap_ns": 800,
            "shorter_ns": 800,
            "overlap_shorter_ppm": 1_000_000,
        }
        delta_tokens, _, phone_delta_metrics = phone_result
        _, cuda_replay_metrics = cuda_result
        prediction, cuda_delta_metrics = live.w6.feed_known_tokens(
            cuda,
            delta_tokens,
            30000,
            self.base.prompt_tokens
            + self.contract.preexisting_committed_tokens
            + service_count,
            self.contract.cuda_delta_chunk,
        )
        cuda_tokens, cuda_cont_metrics = live.w6.continue_from_prediction(
            cuda,
            prediction,
            30000,
            self.base.prompt_tokens
            + self.contract.preexisting_committed_tokens
            + service_count
            + self.contract.phone_delta_tokens,
            self.contract.cuda_continuation_tokens,
        )
        live.w5.remove_group(cuda, self.contract.batch, 30000)
        frontier = [
            history + delta
            for history, delta in zip(histories, delta_tokens)
        ]
        control_tokens, warm_metrics = live.w5.run_replay(
            cuda,
            frontier,
            40000,
            self.contract.cuda_control_chunk,
            self.contract.cuda_continuation_tokens,
        )
        live.w5.remove_group(cuda, self.contract.batch, 40000)
        live.w5.remove_group(phone, self.contract.batch, 10000)

        tx_id = live.transaction_id(self.contract, self.base, self.RUN_ID)
        journal = live.w6.OwnershipJournal(
            journal_path,
            tx_id,
            self.base.model_sha256,
            self.base.prompt_ids,
            self.RUN_ID,
        )
        frontier_sha = live.w5.histories_digest(frontier)
        final = [
            history + tokens
            for history, tokens in zip(frontier, cuda_tokens)
        ]
        final_sha = live.w5.histories_digest(final)
        published = (
            self.contract.preexisting_committed_tokens
            + service_count
            + self.contract.phone_delta_tokens
        )
        states = (
            ("PHONE_FRONTIER", "PHONE", 1, True, True, frontier_sha, published),
            ("CUDA_PREPARED", "PHONE", 1, True, True, frontier_sha, published),
            ("CUDA_COMMITTED", "CUDA", 2, True, True, frontier_sha, published),
            ("PHONE_RELEASED", "CUDA", 2, False, True, frontier_sha, published),
            (
                "CUDA_CONTINUATION",
                "CUDA",
                2,
                False,
                True,
                final_sha,
                published + self.contract.cuda_continuation_tokens,
            ),
            (
                "COMPLETE",
                "NONE",
                3,
                False,
                False,
                final_sha,
                published + self.contract.cuda_continuation_tokens,
            ),
        )
        for phase, owner, epoch, phone_active, cuda_active, digest, count in states:
            journal.append(
                phase=phase,
                owner=owner,
                owner_epoch=epoch,
                published_tokens_per_request=count,
                token_history_sha256=digest,
                phone_active=phone_active,
                cuda_active=cuda_active,
            )
        request_start = service_metrics.batch_timeline[0].started_ns - 1000
        frontier_ns = service_metrics.token_ready_ns[-1]
        report = {
            "base_contract_sha256": self.base.raw_sha256,
            "batch": self.contract.batch,
            "concurrency": concurrency,
            "contract_sha256": self.contract.raw_sha256,
            "corpus_manifest_sha256": self.base.manifest_sha256,
            "corpus_sha256": self.base.corpus_sha256,
            "cuda_ready": {
                "attempts": 3,
                "cuda_ready_ns": connector.ready_ns,
                "max_inter_batch_gap_us": (
                    cold.max_inter_batch_gap_us(service_metrics)
                ),
                "ownership_commit_ns": frontier_ns + 100,
                "phone_first_token_ns": service_metrics.token_ready_ns[0],
                "phone_frontier_ns": frontier_ns,
                "phone_service_tokens": service_count,
                "request_complete_ns": frontier_ns + 200,
                "request_start_ns": request_start,
                "useful_phone_tokens_before_ready": sum(
                    item < connector.ready_ns
                    for item in service_metrics.token_ready_ns
                ),
            },
            "delta_contract_sha256": self.delta.raw_sha256,
            "hellos": {"cuda": self.hello(), "phone": self.hello()},
            "journal": live.w6.journal_summary(journal.entries),
            "metrics": {
                "cuda_continuation": asdict(cuda_cont_metrics),
                "cuda_delta": asdict(cuda_delta_metrics),
                "cuda_replay": asdict(cuda_replay_metrics),
                "cuda_warm_control": asdict(warm_metrics),
                "phone_delta": asdict(phone_delta_metrics),
                "phone_service": cold.service_metrics_value(service_metrics),
            },
            "model_sha256": self.base.model_sha256,
            "physical_gate_sha256": self.gate.raw_sha256,
            "preexisting": {
                "ended_ns": request_start - 1,
                "metrics": cold.service_metrics_value(pre_metrics),
                "preexisting_committed_tokens": (
                    self.contract.preexisting_committed_tokens
                ),
                "started_ns": pre_metrics.batch_timeline[0].started_ns,
                "state_count": self.contract.batch,
            },
            "prompts": list(self.base.prompt_ids),
            "run_id": self.RUN_ID,
            "scheduler_eligible": False,
            "schema": live.SCHEMA,
            "scope": "MECHANICS_ONLY",
            "sequences": [
                {
                    "control_continuation": control_tokens[index],
                    "cuda_continuation": cuda_tokens[index],
                    "final_published_tokens": (
                        preexisting[index]
                        + service[index]
                        + delta_tokens[index]
                        + cuda_tokens[index]
                    ),
                    "phone_delta": delta_tokens[index],
                    "phone_service": service[index],
                    "preexisting_tokens": preexisting[index],
                    "prompt_id": self.base.prompt_ids[index],
                    "prompt_tokens": prompts[index],
                    "sequence_index": index,
                }
                for index in range(self.contract.batch)
            ],
            "state_counts": {
                "cuda_after_completion": 0,
                "cuda_control_released": 0,
                "cuda_prepared": self.contract.batch,
                "phone_after_commit": 0,
                "phone_at_frontier": self.contract.batch,
                "phone_before_promotion": self.contract.batch,
            },
            "status": "LIVE_SESSION_PROMOTION_PASS",
        }
        return report, service, preexisting

    def test_contract_is_frozen(self):
        self.assertEqual(self.contract.batch, 8)
        self.assertEqual(self.contract.preexisting_committed_tokens, 2)
        self.assertEqual(self.contract.min_phone_service_tokens, 1)

    def test_external_validator_has_fail_closed_protocol_error(self):
        self.assertIn(validator.ProtocolError, validator.FAIL_CLOSED_ERRORS)

    def test_r1_contract_is_frozen(self):
        self.assertNotEqual(
            self.contract.raw_sha256,
            self.r1_contract.raw_sha256,
        )
        self.assertEqual(self.r1_contract.phone_delta_tokens, 2)

    def test_teacher_forced_control_consumes_treatment_trace(self):
        histories = [
            prompt + [30000 + seq_id * 2, 30001 + seq_id * 2]
            for seq_id, prompt in enumerate(self.prompts())
        ]
        trace = [
            [40000 + seq_id * 10 + offset for offset in range(4)]
            for seq_id in range(self.r1_contract.batch)
        ]
        client = FakeClient()
        predicted, metrics = trace_control.run_teacher_forced(
            client,
            histories,
            trace,
            50000,
            self.r1_contract.cuda_control_chunk,
        )
        self.assertEqual(len(predicted), self.r1_contract.batch)
        self.assertEqual(len(predicted[0]), 4)
        self.assertEqual(metrics.continuation_batches, 3)
        for seq_id in range(self.r1_contract.batch):
            self.assertEqual(
                client.histories[seq_id],
                histories[seq_id] + trace[seq_id][:-1],
            )

    def test_trace_control_reports_divergence_without_failing(self):
        histories = [
            prompt + [30000 + seq_id * 2, 30001 + seq_id * 2]
            for seq_id, prompt in enumerate(self.prompts())
        ]
        trace = [
            [40000 + seq_id * 10 + offset for offset in range(4)]
            for seq_id in range(self.r1_contract.batch)
        ]
        predicted, metrics = trace_control.run_teacher_forced(
            FakeClient(),
            histories,
            trace,
            50000,
            self.r1_contract.cuda_control_chunk,
        )
        matches = sum(
            actual == expected
            for actual_row, expected_row in zip(predicted, trace)
            for actual, expected in zip(actual_row, expected_row)
        )
        first = metrics.token_ready_ns[0]
        report = {
            "base_contract_sha256": self.base.raw_sha256,
            "batch": self.r1_contract.batch,
            "connector_attempts": 2,
            "contract_sha256": self.r1_contract.raw_sha256,
            "control_mode": trace_control.MODE,
            "cuda_launch_ns": first - 200,
            "cuda_ready_ns": first - 100,
            "first_token_ns": first,
            "greedy_agreement": {
                "all_match": (
                    matches == self.r1_contract.batch * len(trace[0])
                ),
                "matching_tokens": matches,
                "total_tokens": self.r1_contract.batch * len(trace[0]),
            },
            "hello": self.hello(),
            "metrics": cold.service_metrics_value(metrics),
            "model_sha256": self.base.model_sha256,
            "post_start_tokens": len(trace[0]),
            "preexisting_committed_tokens": (
                self.r1_contract.preexisting_committed_tokens
            ),
            "prompts": list(self.base.prompt_ids),
            "request_complete_ns": metrics.token_ready_ns[-1],
            "request_start_ns": first - 300,
            "run_id": self.RUN_ID,
            "scheduler_eligible": False,
            "schema": trace_control.SCHEMA,
            "scope": "MECHANICS_ONLY",
            "sequences": [
                {
                    "predicted_tokens": predicted[index],
                    "preexisting_tokens": histories[index][
                        self.base.prompt_tokens:
                    ],
                    "prompt_id": self.base.prompt_ids[index],
                    "prompt_tokens": histories[index][
                        :self.base.prompt_tokens
                    ],
                    "replayed_tokens": list(trace[index]),
                    "sequence_index": index,
                }
                for index in range(self.r1_contract.batch)
            ],
            "state_count": 0,
            "status": "LIVE_SESSION_SERVER_TRACE_CONTROL_PASS",
        }
        trace_control.validate_report(
            report,
            self.r1_contract,
            self.base,
            self.RUN_ID,
            trace,
        )
        report["sequences"][0]["replayed_tokens"][0] += 1
        with self.assertRaisesRegex(
            live.LivePromotionError,
            "sequence",
        ):
            trace_control.validate_report(
                report,
                self.r1_contract,
                self.base,
                self.RUN_ID,
                trace,
            )

    def test_live_decode_starts_from_preexisting_kv(self):
        phone = FakeClient()
        preexisting, _ = cold.run_fixed_generation(
            phone,
            self.prompts(),
            10000,
            self.contract.phone_prefill_chunk,
            self.contract.preexisting_committed_tokens,
        )
        connector = BoundaryReady(2)
        service, metrics = live.continue_active_until_ready(
            phone,
            [tokens[-1] for tokens in preexisting],
            10000,
            self.base.prompt_tokens
            + self.contract.preexisting_committed_tokens
            - 1,
            1,
            24,
            connector,
        )
        self.assertEqual(len(service[0]), 2)
        self.assertEqual(metrics.history_batches, 0)
        self.assertEqual(metrics.rows, self.contract.batch * 2)

    def test_treatment_report_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = pathlib.Path(directory) / "journal"
            report, _, _ = self.treatment_flow(journal)
            live.validate_report(
                report,
                self.contract,
                self.base,
                self.delta,
                journal,
                self.RUN_ID,
            )

    def test_preexisting_state_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = pathlib.Path(directory) / "journal"
            report, _, _ = self.treatment_flow(journal)
            report["preexisting"]["state_count"] = 0
            with self.assertRaisesRegex(
                live.LivePromotionError,
                "preexisting state",
            ):
                live.validate_report(
                    report,
                    self.contract,
                    self.base,
                    self.delta,
                    journal,
                    self.RUN_ID,
                )

    def test_outer_validator_rejects_failed_internal_exactness(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = pathlib.Path(directory) / "journal"
            treatment, _, _ = self.treatment_flow(journal)
            treatment["sequences"][0]["control_continuation"][0] += 1
            treatment["status"] = "LIVE_SESSION_PROMOTION_FAIL"
            live.validate_report(
                treatment,
                self.contract,
                self.base,
                self.delta,
                journal,
                self.RUN_ID,
            )
            with self.assertRaisesRegex(
                live.LivePromotionError,
                "internal exactness",
            ):
                validator.require_passing_treatment(treatment)

    def test_matched_control_compares_only_post_start_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = pathlib.Path(directory) / "journal"
            treatment, service, preexisting = self.treatment_flow(journal)
            post_start = len(service[0]) + 2 + 8
            histories = [
                prompt + before
                for prompt, before in zip(self.prompts(), preexisting)
            ]
            generated, metrics = cold.run_fixed_generation(
                FakeClient(),
                histories,
                50000,
                self.contract.cuda_control_chunk,
                post_start,
            )
            first = metrics.token_ready_ns[0]
            control_report = {
                "base_contract_sha256": self.base.raw_sha256,
                "batch": self.contract.batch,
                "connector_attempts": 2,
                "contract_sha256": self.contract.raw_sha256,
                "cuda_launch_ns": first - 200,
                "cuda_ready_ns": first - 100,
                "first_token_ns": first,
                "hello": self.hello(),
                "metrics": cold.service_metrics_value(metrics),
                "model_sha256": self.base.model_sha256,
                "post_start_tokens": post_start,
                "preexisting_committed_tokens": (
                    self.contract.preexisting_committed_tokens
                ),
                "prompts": list(self.base.prompt_ids),
                "request_complete_ns": metrics.token_ready_ns[-1],
                "request_start_ns": first - 300,
                "run_id": self.RUN_ID,
                "scheduler_eligible": False,
                "schema": control.SCHEMA,
                "scope": "MECHANICS_ONLY",
                "sequences": [
                    {
                        "generated_tokens": generated[index],
                        "preexisting_tokens": preexisting[index],
                        "prompt_id": self.base.prompt_ids[index],
                        "prompt_tokens": self.prompts()[index],
                        "sequence_index": index,
                    }
                    for index in range(self.contract.batch)
                ],
                "state_count": 0,
                "status": "LIVE_SESSION_SERVER_CONTROL_PASS",
            }
            control.validate_report(
                control_report,
                self.contract,
                self.base,
                self.RUN_ID,
                post_start,
            )
            control_report["metrics"]["batch_timeline"][0]["rows"] += 1
            with self.assertRaisesRegex(
                live.LivePromotionError,
                "invalid timeline",
            ):
                control.validate_report(
                    control_report,
                    self.contract,
                    self.base,
                    self.RUN_ID,
                    post_start,
                )
            control_report["metrics"]["batch_timeline"][0]["rows"] -= 1
            validator.validate_matched_tokens(treatment, control_report)
            control_report["sequences"][0]["generated_tokens"][0] += 1
            with self.assertRaisesRegex(
                live.LivePromotionError,
                "token mismatch",
            ):
                validator.validate_matched_tokens(treatment, control_report)


if __name__ == "__main__":
    unittest.main()
