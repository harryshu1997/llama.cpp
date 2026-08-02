#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import tempfile
import time
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest import mock


HERE = pathlib.Path(__file__).parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROMOTION = load_module(
    "phone_cuda_cold_promotion_probe",
    "phone_cuda_cold_promotion_probe.py",
)
CONTROL = load_module("cuda_cold_control_probe", "cuda_cold_control_probe.py")
PREPARATION = load_module("prepare_phone_route", "prepare_phone_route.py")
VALIDATOR = load_module("validate_cold_promotion", "validate_cold_promotion.py")


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
                raise PROMOTION.ProtocolError("noncontiguous fake history")
            history.append(row.token)
            results.append(PROMOTION.w6.BatchResult(
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


class DelayedReady:
    def __init__(self, ready_check):
        self.ready_check = ready_check
        self.checks = 0

    @staticmethod
    def raise_if_failed():
        return None

    def is_ready(self):
        self.checks += 1
        return self.checks >= self.ready_check


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


class ColdPromotionTests(unittest.TestCase):
    RUN_ID = "a" * 64

    @classmethod
    def setUpClass(cls):
        cls.base = PROMOTION.w5.load_contract(PROMOTION.DEFAULT_BASE_CONTRACT)
        cls.delta = PROMOTION.w6.load_contract(
            PROMOTION.DEFAULT_DELTA_CONTRACT,
            cls.base,
        )
        cls.gate = VALIDATOR.physical.load_physical_gate(
            PROMOTION.DEFAULT_PHYSICAL_GATE,
            cls.delta,
            cls.base,
        )
        cls.contract = PROMOTION.load_contract(
            PROMOTION.DEFAULT_CONTRACT,
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

    def preparation_report(self):
        prompts = self.prompts()
        tokens, metrics = PROMOTION.run_fixed_generation(
            FakeClient(),
            prompts,
            70000,
            self.contract.phone_prefill_chunk,
            self.contract.warmup_generated_tokens,
        )
        return {
            "base_contract_sha256": self.base.raw_sha256,
            "batch": self.contract.batch,
            "contract_sha256": self.contract.raw_sha256,
            "ended_ns": 200,
            "hello": self.hello(),
            "metrics": PROMOTION.service_metrics_value(metrics),
            "model_sha256": self.base.model_sha256,
            "prompts": list(self.base.prompt_ids),
            "request_state_reset": True,
            "run_id": self.RUN_ID,
            "scheduler_eligible": False,
            "schema": PREPARATION.SCHEMA,
            "scope": "MECHANICS_ONLY",
            "sequences": [
                {
                    "generated_tokens": tokens[index],
                    "prompt_id": self.base.prompt_ids[index],
                    "prompt_tokens": prompts[index],
                    "sequence_index": index,
                }
                for index in range(self.contract.batch)
            ],
            "session_end": "DETACH",
            "started_ns": 100,
            "state_count": 0,
            "status": "PHONE_ROUTE_PREPARATION_PASS",
            "warmup_generated_tokens": self.contract.warmup_generated_tokens,
        }

    def session_certificate(
        self,
        gate_name,
        boot_id,
        session_id,
        session_end,
        reset_applied,
        steps_session,
        steps_total,
        worker_pid,
        worker_boot_nonce,
    ):
        spec = self.gate.placements[gate_name]
        return {
            "compute_by_op_and_buffer": {
                "MUL_MAT": {spec["primary_buffer"]: 1},
            },
            "device_boot_id": boot_id,
            "expected_backend": spec["expected_backend"],
            "layer_end": spec["layer_end"],
            "layer_start": spec["layer_start"],
            "missing_buffer_compute_nodes": 0,
            "n_layer": self.gate.n_layer,
            "placement_status": "SCHEDULED_PLACEMENT_OK",
            "proto_version": 2,
            "reset_applied": reset_applied,
            "schema": VALIDATOR.physical.SESSION_SCHEMA,
            "session_end": session_end,
            "session_id": session_id,
            "steps_session": steps_session,
            "steps_total": steps_total,
            "worker_boot_nonce": worker_boot_nonce,
            "worker_pid": worker_pid,
        }

    def write_session_log(self, path, certificates):
        raw = b"".join(
            b"SESSIONCERT " + PROMOTION.w6.canonical(value)
            for value in certificates
        )
        path.write_bytes(raw)

    def treatment_flow(self, journal_path):
        prompts = self.prompts()
        phone = FakeClient()
        cuda = FakeClient()
        phone_tokens, service_metrics = PROMOTION.serve_until_ready(
            phone,
            prompts,
            10000,
            self.contract.phone_prefill_chunk,
            self.contract.min_phone_service_tokens,
            self.contract.max_phone_service_tokens,
            DelayedReady(3),
        )
        service_count = len(phone_tokens[0])
        service_histories = [
            prompt + tokens
            for prompt, tokens in zip(prompts, phone_tokens)
        ]
        delta_tokens, _, phone_delta_metrics = PROMOTION.w6.advance_active(
            phone,
            [tokens[-1] for tokens in phone_tokens],
            10000,
            self.base.prompt_tokens + service_count - 1,
            self.contract.phone_delta_tokens,
        )
        _, cuda_replay_metrics = PROMOTION.w6.replay_history_only(
            cuda,
            service_histories,
            30000,
            self.contract.cuda_replay_chunk,
        )
        prediction, cuda_delta_metrics = PROMOTION.w6.feed_known_tokens(
            cuda,
            delta_tokens,
            30000,
            self.base.prompt_tokens + service_count,
            self.contract.cuda_delta_chunk,
        )
        cuda_tokens, cuda_cont_metrics = PROMOTION.w6.continue_from_prediction(
            cuda,
            prediction,
            30000,
            self.base.prompt_tokens
            + service_count
            + self.contract.phone_delta_tokens,
            self.contract.cuda_continuation_tokens,
        )
        PROMOTION.w5.remove_group(cuda, self.contract.batch, 30000)
        frontier = [
            history + tokens
            for history, tokens in zip(service_histories, delta_tokens)
        ]
        control_tokens, control_metrics = PROMOTION.w5.run_replay(
            cuda,
            frontier,
            40000,
            self.contract.cuda_control_chunk,
            self.contract.cuda_continuation_tokens,
        )
        PROMOTION.w5.remove_group(cuda, self.contract.batch, 40000)

        tx_id = PROMOTION.transaction_id(
            self.contract,
            self.base,
            self.RUN_ID,
        )
        journal = PROMOTION.w6.OwnershipJournal(
            journal_path,
            tx_id,
            self.base.model_sha256,
            self.base.prompt_ids,
            self.RUN_ID,
        )
        frontier_sha = PROMOTION.w5.histories_digest(frontier)
        final = [
            history + tokens
            for history, tokens in zip(frontier, cuda_tokens)
        ]
        final_sha = PROMOTION.w5.histories_digest(final)
        published = service_count + self.contract.phone_delta_tokens
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
        ready_ns = service_metrics.token_ready_ns[-2] + 1
        request_start_ns = service_metrics.batch_timeline[0].started_ns - 1000
        frontier_ns = service_metrics.token_ready_ns[-1]
        report = {
            "base_contract_sha256": self.base.raw_sha256,
            "batch": self.contract.batch,
            "concurrency": {
                "cuda_ended_ns": 900,
                "cuda_started_ns": 100,
                "cuda_wall_ns": 800,
                "overlap_ns": 800,
                "overlap_shorter_ppm": 1_000_000,
                "phone_ended_ns": 1100,
                "phone_started_ns": 100,
                "phone_wall_ns": 1000,
                "shorter_ns": 800,
            },
            "contract_sha256": self.contract.raw_sha256,
            "corpus_manifest_sha256": self.base.manifest_sha256,
            "corpus_sha256": self.base.corpus_sha256,
            "cuda_ready": {
                "attempts": 3,
                "cuda_ready_ns": ready_ns,
                "max_inter_batch_gap_us": (
                    PROMOTION.max_inter_batch_gap_us(service_metrics)
                ),
                "ownership_commit_ns": frontier_ns + 100,
                "phone_first_token_ns": service_metrics.token_ready_ns[0],
                "phone_frontier_ns": frontier_ns,
                "phone_service_tokens": service_count,
                "request_complete_ns": frontier_ns + 200,
                "request_start_ns": request_start_ns,
                "useful_phone_tokens_before_ready": sum(
                    item < ready_ns
                    for item in service_metrics.token_ready_ns
                ),
            },
            "delta_contract_sha256": self.delta.raw_sha256,
            "hellos": {"cuda": self.hello(), "phone": self.hello()},
            "journal": PROMOTION.w6.journal_summary(journal.entries),
            "metrics": {
                "cuda_continuation": asdict(cuda_cont_metrics),
                "cuda_delta": asdict(cuda_delta_metrics),
                "cuda_replay": asdict(cuda_replay_metrics),
                "cuda_warm_control": asdict(control_metrics),
                "phone_delta": asdict(phone_delta_metrics),
                "phone_service": PROMOTION.service_metrics_value(
                    service_metrics
                ),
            },
            "model_sha256": self.base.model_sha256,
            "physical_gate_sha256": self.gate.raw_sha256,
            "prompts": list(self.base.prompt_ids),
            "run_id": self.RUN_ID,
            "scheduler_eligible": False,
            "schema": PROMOTION.SCHEMA,
            "scope": "MECHANICS_ONLY",
            "sequences": [
                {
                    "control_continuation": control_tokens[index],
                    "cuda_continuation": cuda_tokens[index],
                    "final_published_tokens": (
                        phone_tokens[index]
                        + delta_tokens[index]
                        + cuda_tokens[index]
                    ),
                    "phone_delta": delta_tokens[index],
                    "phone_service": phone_tokens[index],
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
            },
            "status": "COLD_PROMOTION_TREATMENT_PASS",
        }
        return report, service_metrics, control_tokens

    def test_contract_is_frozen_and_bounded(self):
        self.assertEqual(self.contract.min_phone_service_tokens, 4)
        self.assertEqual(self.contract.max_phone_service_tokens, 24)
        self.assertFalse(
            PROMOTION.json.loads(PROMOTION.DEFAULT_CONTRACT.read_bytes())[
                "scheduler_eligible_on_pass"
            ]
        )
        self.assertEqual(self.contract.warmup_generated_tokens, 2)

    def test_preparation_report_validates_and_requires_reset(self):
        report = self.preparation_report()
        PREPARATION.validate_report(
            report,
            self.contract,
            self.base,
            self.RUN_ID,
        )
        report["request_state_reset"] = False
        with self.assertRaisesRegex(
            PROMOTION.PromotionError,
            "execution contract",
        ):
            PREPARATION.validate_report(
                report,
                self.contract,
                self.base,
                self.RUN_ID,
            )

    def test_dynamic_phone_service_stops_at_ready_boundary(self):
        client = FakeClient()
        tokens, metrics = PROMOTION.serve_until_ready(
            client,
            self.prompts(),
            10000,
            self.contract.phone_prefill_chunk,
            4,
            24,
            DelayedReady(3),
        )
        self.assertEqual(len(tokens[0]), 4)
        self.assertEqual(metrics.continuation_batches, 3)
        self.assertEqual(
            metrics.rows,
            self.contract.batch * (self.base.prompt_tokens + 3),
        )

    def test_dynamic_service_finishes_a_boundary_after_readiness(self):
        connector = BoundaryReady(2)
        tokens, metrics = PROMOTION.serve_until_ready(
            FakeClient(),
            self.prompts(),
            10000,
            self.contract.phone_prefill_chunk,
            1,
            24,
            connector,
        )
        self.assertEqual(len(tokens[0]), 3)
        self.assertLessEqual(connector.ready_ns, metrics.token_ready_ns[-1])

    def test_connector_take_waits_beyond_one_second(self):
        deadline = time.monotonic() + 1.05
        fake_client = SimpleNamespace(hello=lambda: PROMOTION.Hello(**self.hello()))

        def delayed_connect(*_args):
            if time.monotonic() < deadline:
                raise ConnectionRefusedError
            return fake_client

        connector = PROMOTION.ReadyConnector(("127.0.0.1", 1), 2.0, 10000)
        with mock.patch.object(
            PROMOTION.StageV3Client,
            "connect",
            side_effect=delayed_connect,
        ):
            connector.start()
            result = connector.take()
        self.assertIs(result.client, fake_client)
        connector.close()

    def test_fixed_generation_matches_existing_replay(self):
        prompts = self.prompts()
        fixed, fixed_metrics = PROMOTION.run_fixed_generation(
            FakeClient(),
            prompts,
            10000,
            2,
            8,
        )
        replay, replay_metrics = PROMOTION.w5.run_replay(
            FakeClient(),
            prompts,
            20000,
            2,
            8,
        )
        self.assertEqual(fixed, replay)
        self.assertEqual(fixed_metrics.rows, replay_metrics.rows)

    def test_treatment_report_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = pathlib.Path(directory) / "journal"
            report, _, _ = self.treatment_flow(journal)
            PROMOTION.validate_report(
                report,
                self.contract,
                self.base,
                self.delta,
                journal,
                self.RUN_ID,
            )

    def test_treatment_token_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = pathlib.Path(directory) / "journal"
            report, _, _ = self.treatment_flow(journal)
            report["sequences"][0]["phone_delta"][0] += 1
            with self.assertRaisesRegex(
                PROMOTION.PromotionError,
                "publication",
            ):
                PROMOTION.validate_report(
                    report,
                    self.contract,
                    self.base,
                    self.delta,
                    journal,
                    self.RUN_ID,
                )

    def test_treatment_concurrency_shape_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = pathlib.Path(directory) / "journal"
            report, _, _ = self.treatment_flow(journal)
            del report["concurrency"]["shorter_ns"]
            with self.assertRaisesRegex(
                PROMOTION.w6.DeltaError,
                "treatment.concurrency",
            ):
                PROMOTION.validate_report(
                    report,
                    self.contract,
                    self.base,
                    self.delta,
                    journal,
                    self.RUN_ID,
                )

    def test_control_report_validates(self):
        prompts = self.prompts()
        total = 16
        tokens, metrics = PROMOTION.run_fixed_generation(
            FakeClient(),
            prompts,
            50000,
            self.contract.cuda_control_chunk,
            total,
        )
        first = metrics.token_ready_ns[0]
        report = {
            "base_contract_sha256": self.base.raw_sha256,
            "batch": self.contract.batch,
            "connector_attempts": 4,
            "contract_sha256": self.contract.raw_sha256,
            "cuda_launch_ns": first - 200,
            "cuda_ready_ns": first - 100,
            "first_token_ns": first,
            "hello": self.hello(),
            "metrics": PROMOTION.service_metrics_value(metrics),
            "model_sha256": self.base.model_sha256,
            "prompts": list(self.base.prompt_ids),
            "request_complete_ns": metrics.token_ready_ns[-1],
            "request_start_ns": first - 300,
            "run_id": self.RUN_ID,
            "scheduler_eligible": False,
            "schema": CONTROL.SCHEMA,
            "scope": "MECHANICS_ONLY",
            "sequences": [
                {
                    "generated_tokens": tokens[index],
                    "prompt_id": self.base.prompt_ids[index],
                    "prompt_tokens": prompts[index],
                    "sequence_index": index,
                }
                for index in range(self.contract.batch)
            ],
            "state_count": 0,
            "status": "COLD_SERVER_QUEUE_CONTROL_PASS",
            "total_generated_tokens": total,
        }
        CONTROL.validate_report(
            report,
            self.contract,
            self.base,
            self.RUN_ID,
            total,
        )
        report["cuda_ready_ns"] = report["request_start_ns"] - 1
        with self.assertRaisesRegex(
            PROMOTION.PromotionError,
            "timing order",
        ):
            CONTROL.validate_report(
                report,
                self.contract,
                self.base,
                self.RUN_ID,
                total,
            )

    def test_matched_control_requires_exact_generated_tokens(self):
        treatment = {
            "sequences": [
                {
                    "final_published_tokens": [1, 2, 3],
                    "prompt_tokens": [10, 11],
                },
            ],
        }
        control = {
            "sequences": [
                {
                    "generated_tokens": [1, 2, 3],
                    "prompt_tokens": [10, 11],
                },
            ],
        }
        VALIDATOR.validate_matched_tokens(treatment, control)
        control["sequences"][0]["generated_tokens"][1] = 9
        with self.assertRaisesRegex(
            PROMOTION.PromotionError,
            "token mismatch",
        ):
            VALIDATOR.validate_matched_tokens(treatment, control)

    def test_placement_requires_same_resident_phone_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            journal = root / "journal"
            treatment, _, _ = self.treatment_flow(journal)
            preparation_report = self.preparation_report()
            warmup_steps = preparation_report["metrics"]["rows"]
            phone_steps = (
                treatment["metrics"]["phone_service"]["rows"]
                + treatment["metrics"]["phone_delta"]["rows"]
            )
            cuda_steps = sum(
                treatment["metrics"][name]["rows"]
                for name in (
                    "cuda_continuation",
                    "cuda_delta",
                    "cuda_replay",
                    "cuda_warm_control",
                )
            )
            control_report = {"metrics": {"rows": 184}}
            context = {
                "cuda": {"boot_id": "cuda-boot"},
                "op12": {"boot_id": "op12-boot"},
                "op15": {"boot_id": "op15-boot"},
            }
            paths = {}
            for index, name in enumerate(("op12", "op15"), 1):
                path = root / f"{name}.log"
                self.write_session_log(path, [
                    self.session_certificate(
                        name,
                        context[name]["boot_id"],
                        1,
                        "DETACH",
                        True,
                        warmup_steps,
                        warmup_steps,
                        index,
                        f"{index:016x}",
                    ),
                    self.session_certificate(
                        name,
                        context[name]["boot_id"],
                        2,
                        "STOP",
                        False,
                        phone_steps,
                        warmup_steps + phone_steps,
                        index,
                        f"{index:016x}",
                    ),
                ])
                paths[name] = path
            for phase, steps, pid_base in (
                ("treatment", cuda_steps, 100),
                ("control", control_report["metrics"]["rows"], 200),
            ):
                for offset, role in enumerate(("head", "tail")):
                    gate_name = f"cuda_{role}"
                    path_name = f"{phase}_{gate_name}"
                    path = root / f"{path_name}.log"
                    pid = pid_base + offset
                    self.write_session_log(path, [
                        self.session_certificate(
                            gate_name,
                            context["cuda"]["boot_id"],
                            1,
                            "STOP",
                            False,
                            steps,
                            steps,
                            pid,
                            f"{pid:016x}",
                        ),
                    ])
                    paths[path_name] = path
            VALIDATOR.validate_placement_sessions(
                paths=paths,
                context=context,
                gate=self.gate,
                preparation_report=preparation_report,
                treatment=treatment,
                control_report=control_report,
            )
            op15 = paths["op15"]
            certificates, _ = VALIDATOR.parse_session_certificates(
                op15,
                "op15",
            )
            certificates[1]["worker_boot_nonce"] = "f" * 16
            self.write_session_log(op15, certificates)
            with self.assertRaisesRegex(
                PROMOTION.PromotionError,
                "worker was not resident",
            ):
                VALIDATOR.validate_placement_sessions(
                    paths=paths,
                    context=context,
                    gate=self.gate,
                    preparation_report=preparation_report,
                    treatment=treatment,
                    control_report=control_report,
                )


if __name__ == "__main__":
    unittest.main()
