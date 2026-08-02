#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import tempfile
import time
import unittest
from dataclasses import asdict
from types import SimpleNamespace


PATH = pathlib.Path(__file__).parents[1] / "phone_cuda_delta_probe.py"
SPEC = importlib.util.spec_from_file_location("phone_cuda_delta_probe", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)

VALIDATOR_PATH = pathlib.Path(__file__).parents[1] / "validate_phone_cuda_delta.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_phone_cuda_delta",
    VALIDATOR_PATH,
)
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
assert VALIDATOR_SPEC.loader is not None
sys.modules[VALIDATOR_SPEC.name] = VALIDATOR
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)


class FakeClient:
    def __init__(self):
        self.histories = {}

    @staticmethod
    def predict(history):
        return (
            sum((index + 1) * token for index, token in enumerate(history))
            + 17
        ) % 50000

    def batch(self, rows):
        results = []
        for row in rows:
            history = self.histories.setdefault(row.seq_id, [])
            if row.position != len(history):
                raise MOD.ProtocolError("noncontiguous fake history")
            history.append(row.token)
            results.append(MOD.BatchResult(
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
        if seq_id not in self.histories:
            raise MOD.ProtocolError("missing fake sequence")
        del self.histories[seq_id]
        return SimpleNamespace(
            active_sequences=len(self.histories),
            max_streams=64,
            draining=False,
        )


class FailingJournal:
    def append(self, **fields):
        del fields
        raise MOD.DeltaError("injected journal failure")


class PhoneCudaDeltaTests(unittest.TestCase):
    RUN_ID = "a" * 64

    @classmethod
    def setUpClass(cls):
        cls.base = MOD.w5.load_contract(MOD.DEFAULT_BASE_CONTRACT)
        cls.contract = MOD.load_contract(MOD.DEFAULT_CONTRACT, cls.base)
        cls.physical_gate = VALIDATOR.load_physical_gate(
            pathlib.Path(__file__).parents[1] / "W6_PHYSICAL_GATE.json",
            cls.contract,
            cls.base,
        )

    def prompts(self):
        return [
            [seq_id * 100 + position for position in range(self.base.prompt_tokens)]
            for seq_id in range(self.contract.batch)
        ]

    def fake_flow(self):
        phone = FakeClient()
        cuda = FakeClient()
        prompts = self.prompts()
        snapshot, snapshot_metrics = MOD.w5.run_replay(
            phone,
            prompts,
            10000,
            self.base.phone_prefill_chunk,
            self.contract.phone_snapshot_tokens,
        )
        delta_tokens, _, phone_delta_metrics = MOD.advance_active(
            phone,
            [tokens[-1] for tokens in snapshot],
            10000,
            self.base.prompt_tokens + self.contract.phone_snapshot_tokens - 1,
            self.contract.phone_delta_tokens,
        )
        snapshot_histories = [
            prompt + tokens
            for prompt, tokens in zip(prompts, snapshot)
        ]
        _, cuda_snapshot_metrics = MOD.replay_history_only(
            cuda,
            snapshot_histories,
            30000,
            self.contract.cuda_snapshot_chunk,
        )
        prediction, cuda_delta_metrics = MOD.feed_known_tokens(
            cuda,
            delta_tokens,
            30000,
            self.base.prompt_tokens + self.contract.phone_snapshot_tokens,
            self.contract.cuda_delta_chunk,
        )
        cuda_tokens, cuda_continuation_metrics = MOD.continue_from_prediction(
            cuda,
            prediction,
            30000,
            self.base.prompt_tokens
            + self.contract.phone_snapshot_tokens
            + self.contract.phone_delta_tokens,
            self.contract.cuda_continuation_tokens,
        )
        MOD.w5.remove_group(cuda, self.contract.batch, 30000)
        frontier = [
            prompt + snap + delta
            for prompt, snap, delta in zip(prompts, snapshot, delta_tokens)
        ]
        control, control_metrics = MOD.w5.run_replay(
            cuda,
            frontier,
            40000,
            self.contract.cuda_control_chunk,
            self.contract.cuda_continuation_tokens,
        )
        MOD.w5.remove_group(cuda, self.contract.batch, 40000)
        return {
            "control": control,
            "control_metrics": control_metrics,
            "cuda": cuda_tokens,
            "cuda_continuation_metrics": cuda_continuation_metrics,
            "cuda_delta_metrics": cuda_delta_metrics,
            "cuda_snapshot_metrics": cuda_snapshot_metrics,
            "delta": delta_tokens,
            "frontier": frontier,
            "phone": phone,
            "phone_delta_metrics": phone_delta_metrics,
            "prompts": prompts,
            "snapshot": snapshot,
            "snapshot_metrics": snapshot_metrics,
        }

    def append_journal(self, path, flow):
        tx_id = MOD.transaction_id(
            self.contract,
            self.base,
            self.RUN_ID,
        )
        journal = MOD.OwnershipJournal(
            path,
            tx_id,
            self.base.model_sha256,
            self.base.prompt_ids,
            self.RUN_ID,
        )
        frontier_sha = MOD.w5.histories_digest(flow["frontier"])
        final = [
            frontier + continuation
            for frontier, continuation in zip(
                flow["frontier"],
                flow["cuda"],
            )
        ]
        final_sha = MOD.w5.histories_digest(final)
        published = (
            self.contract.phone_snapshot_tokens
            + self.contract.phone_delta_tokens
        )
        journal.append(
            phase="PHONE_FRONTIER",
            owner="PHONE",
            owner_epoch=1,
            published_tokens_per_request=published,
            token_history_sha256=frontier_sha,
            phone_active=True,
            cuda_active=True,
        )
        journal.append(
            phase="CUDA_PREPARED",
            owner="PHONE",
            owner_epoch=1,
            published_tokens_per_request=published,
            token_history_sha256=frontier_sha,
            phone_active=True,
            cuda_active=True,
        )
        journal.append(
            phase="CUDA_COMMITTED",
            owner="CUDA",
            owner_epoch=2,
            published_tokens_per_request=published,
            token_history_sha256=frontier_sha,
            phone_active=True,
            cuda_active=True,
        )
        journal.append(
            phase="PHONE_RELEASED",
            owner="CUDA",
            owner_epoch=2,
            published_tokens_per_request=published,
            token_history_sha256=frontier_sha,
            phone_active=False,
            cuda_active=True,
        )
        journal.append(
            phase="CUDA_CONTINUATION",
            owner="CUDA",
            owner_epoch=2,
            published_tokens_per_request=(
                published + self.contract.cuda_continuation_tokens
            ),
            token_history_sha256=final_sha,
            phone_active=False,
            cuda_active=True,
        )
        journal.append(
            phase="COMPLETE",
            owner="NONE",
            owner_epoch=3,
            published_tokens_per_request=(
                published + self.contract.cuda_continuation_tokens
            ),
            token_history_sha256=final_sha,
            phone_active=False,
            cuda_active=False,
        )
        return journal

    def valid_report(self, journal_path, flow):
        entries = self.append_journal(journal_path, flow).entries
        hello = {
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
        sequences = [
            {
                "control_continuation": flow["control"][index],
                "cuda_continuation": flow["cuda"][index],
                "final_published_tokens": (
                    flow["snapshot"][index]
                    + flow["delta"][index]
                    + flow["cuda"][index]
                ),
                "phone_delta": flow["delta"][index],
                "phone_published_tokens": (
                    flow["snapshot"][index] + flow["delta"][index]
                ),
                "phone_snapshot": flow["snapshot"][index],
                "prompt_id": self.base.prompt_ids[index],
                "prompt_tokens": flow["prompts"][index],
                "sequence_index": index,
            }
            for index in range(self.contract.batch)
        ]
        return {
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
            "hellos": {"cuda": dict(hello), "phone": dict(hello)},
            "journal": MOD.journal_summary(entries),
            "metrics": {
                "cuda_continuation": asdict(
                    flow["cuda_continuation_metrics"]
                ),
                "cuda_control": asdict(flow["control_metrics"]),
                "cuda_delta": asdict(flow["cuda_delta_metrics"]),
                "cuda_snapshot": asdict(flow["cuda_snapshot_metrics"]),
                "phone_delta": asdict(flow["phone_delta_metrics"]),
                "phone_snapshot": asdict(flow["snapshot_metrics"]),
            },
            "model_sha256": self.base.model_sha256,
            "phone_delta_tokens": self.contract.phone_delta_tokens,
            "phone_snapshot_tokens": self.contract.phone_snapshot_tokens,
            "prompts": list(self.base.prompt_ids),
            "run_id": self.RUN_ID,
            "scheduler_eligible": False,
            "schema": MOD.SCHEMA,
            "scope": "MECHANICS_ONLY",
            "sequences": sequences,
            "state_counts": {
                "cuda_after_completion": 0,
                "cuda_after_snapshot": self.contract.batch,
                "cuda_control_released": 0,
                "cuda_prepared": self.contract.batch,
                "phone_after_commit": 0,
                "phone_at_frontier": self.contract.batch,
            },
            "status": "CONCURRENT_DELTA_MECHANICS_PASS",
        }

    def valid_run_context(self):
        sources = {
            "async_pipeline.py": (
                VALIDATOR.HERE.parent
                / "s22_slo_overlap_pipeline"
                / "async_pipeline.py"
            ),
            "phone_cuda_delta_probe.py": (
                VALIDATOR.HERE / "phone_cuda_delta_probe.py"
            ),
            "phone_cuda_handoff_probe.py": (
                VALIDATOR.HERE / "phone_cuda_handoff_probe.py"
            ),
            "qwen25_quality_probe.py": (
                VALIDATOR.HERE / "qwen25_quality_probe.py"
            ),
            "run_w6_delta_gate.sh": (
                VALIDATOR.HERE / "run_w6_delta_gate.sh"
            ),
            "stage_v3_client.py": (
                VALIDATOR.HERE.parent
                / "s22_slo_overlap_pipeline"
                / "stage_v3_client.py"
            ),
            "validate_phone_cuda_delta.py": (
                VALIDATOR.HERE / "validate_phone_cuda_delta.py"
            ),
        }
        return {
            "acquisition_unix_s": 1,
            "base_contract_sha256": self.base.raw_sha256,
            "base_git_commit": "0" * 40,
            "contract_sha256": self.contract.raw_sha256,
            "cuda": {
                "boot_id": "boot-cuda",
                "device": "CUDA0",
                "head_port": 41410,
                "relay_port": 41412,
                "relay_sha256": self.physical_gate.artifacts[
                    "host_relay_sha256"
                ],
                "tail_port": 41411,
                "worker_sha256": self.physical_gate.artifacts[
                    "host_worker_sha256"
                ],
            },
            "model_sha256": self.base.model_sha256,
            "op12": {
                "adb_target": "op12",
                "boot_id": "boot12",
                "layers": [30, 48],
                "shard_sha256": self.physical_gate.artifacts[
                    "op12_shard_sha256"
                ],
                "wifi": "127.0.0.12",
                "worker_sha256": self.physical_gate.artifacts[
                    "phone_worker_sha256"
                ],
            },
            "op15": {
                "adb_target": "op15",
                "boot_id": "boot15",
                "layers": [0, 30],
                "relay_sha256": self.physical_gate.artifacts[
                    "op15_relay_sha256"
                ],
                "shard_sha256": self.physical_gate.artifacts[
                    "op15_shard_sha256"
                ],
                "wifi": "127.0.0.15",
                "worker_sha256": self.physical_gate.artifacts[
                    "phone_worker_sha256"
                ],
            },
            "physical_gate_sha256": self.physical_gate.raw_sha256,
            "run_id": self.RUN_ID,
            "schema": "s39-phone-cuda-delta-context-v1",
            "sources": {
                name: MOD.sha256(path.read_bytes())
                for name, path in sources.items()
            },
        }

    def write_session_log(
        self,
        path,
        name,
        report,
        *,
        core_buffer=None,
        step_delta=0,
    ):
        spec = self.physical_gate.placements[name]
        phone_steps = (
            report["metrics"]["phone_snapshot"]["rows"]
            + report["metrics"]["phone_delta"]["rows"]
        )
        cuda_steps = sum(
            report["metrics"][metric]["rows"]
            for metric in (
                "cuda_snapshot",
                "cuda_delta",
                "cuda_continuation",
                "cuda_control",
            )
        )
        steps = phone_steps if name.startswith("op") else cuda_steps
        boot_id = {
            "cuda_head": "boot-cuda",
            "cuda_tail": "boot-cuda",
            "op12": "boot12",
            "op15": "boot15",
        }[name]
        core = core_buffer or spec["primary_buffer"]
        certificate = {
            "compute_by_op_and_buffer": {
                "GET_ROWS": {spec["primary_buffer"]: 1},
                "MUL_MAT": {core: 4},
            },
            "device_boot_id": boot_id,
            "expected_backend": spec["expected_backend"],
            "layer_end": spec["layer_end"],
            "layer_start": spec["layer_start"],
            "missing_buffer_compute_nodes": 0,
            "n_layer": self.physical_gate.n_layer,
            "placement_status": "SCHEDULED_PLACEMENT_OK",
            "proto_version": 2,
            "reset_applied": False,
            "schema": VALIDATOR.SESSION_SCHEMA,
            "session_end": "STOP",
            "session_id": 1,
            "steps_session": steps + step_delta,
            "steps_total": steps + step_delta,
            "worker_boot_nonce": "1" * 16,
            "worker_pid": 42,
        }
        path.write_text(
            "worker output\nSESSIONCERT "
            + MOD.json.dumps(certificate, separators=(",", ":"))
            + "\n",
            encoding="ascii",
        )

    def placement_logs(self, directory, report, **overrides):
        paths = {}
        for name in self.physical_gate.placements:
            path = directory / f"{name}.log"
            self.write_session_log(
                path,
                name,
                report,
                **overrides.get(name, {}),
            )
            paths[name] = path
        return paths

    def test_contract_is_frozen_and_scheduler_ineligible(self):
        self.assertEqual(self.contract.min_overlap_shorter_ppm, 800000)
        value = MOD.json.loads(MOD.DEFAULT_CONTRACT.read_bytes())
        self.assertFalse(value["scheduler_eligible_on_pass"])

    def test_delta_catchup_matches_full_history_control(self):
        flow = self.fake_flow()
        self.assertEqual(flow["cuda"], flow["control"])

    def test_concurrent_legs_overlap(self):
        def phone():
            time.sleep(0.05)
            return "phone"

        def cuda():
            time.sleep(0.03)
            return "cuda"

        phone_value, cuda_value, timing = MOD.run_concurrently(
            phone,
            cuda,
            1.0,
        )
        self.assertEqual((phone_value, cuda_value), ("phone", "cuda"))
        self.assertGreaterEqual(
            timing["overlap_shorter_ppm"],
            self.contract.min_overlap_shorter_ppm,
        )

    def test_commit_record_failure_does_not_remove_phone(self):
        removed = []
        with self.assertRaisesRegex(MOD.DeltaError, "injected"):
            MOD.commit_cutover(
                FailingJournal(),
                published_tokens=6,
                history_sha256="1" * 64,
                remove_phone=lambda: removed.append(True),
            )
        self.assertEqual(removed, [])

    def test_remove_failure_leaves_cuda_as_durable_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            path = pathlib.Path(directory) / "journal"
            tx_id = MOD.transaction_id(
                self.contract,
                self.base,
                self.RUN_ID,
            )
            journal = MOD.OwnershipJournal(
                path,
                tx_id,
                self.base.model_sha256,
                self.base.prompt_ids,
                self.RUN_ID,
            )
            frontier_sha = MOD.w5.histories_digest(flow["frontier"])
            published = (
                self.contract.phone_snapshot_tokens
                + self.contract.phone_delta_tokens
            )
            for phase in ("PHONE_FRONTIER", "CUDA_PREPARED"):
                journal.append(
                    phase=phase,
                    owner="PHONE",
                    owner_epoch=1,
                    published_tokens_per_request=published,
                    token_history_sha256=frontier_sha,
                    phone_active=True,
                    cuda_active=True,
                )

            def fail_remove():
                raise MOD.DeltaError("injected remove failure")

            with self.assertRaisesRegex(MOD.DeltaError, "remove failure"):
                MOD.commit_cutover(
                    journal,
                    published_tokens=published,
                    history_sha256=frontier_sha,
                    remove_phone=fail_remove,
                )
            self.assertEqual(journal.entries[-1].value["owner"], "CUDA")
            self.assertEqual(
                journal.entries[-1].value["phase"],
                "CUDA_COMMITTED",
            )

    def test_journal_rejects_out_of_order_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = MOD.OwnershipJournal(
                pathlib.Path(directory) / "journal",
                "b" * 64,
                self.base.model_sha256,
                self.base.prompt_ids,
                self.RUN_ID,
            )
            with self.assertRaisesRegex(
                MOD.DeltaError,
                "invalid state transition",
            ):
                journal.append(
                    phase="CUDA_PREPARED",
                    owner="PHONE",
                    owner_epoch=1,
                    published_tokens_per_request=6,
                    token_history_sha256="c" * 64,
                    phone_active=True,
                    cuda_active=True,
                )

    def test_journal_rejects_append_after_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            journal = self.append_journal(
                pathlib.Path(directory) / "journal",
                flow,
            )
            with self.assertRaisesRegex(
                MOD.DeltaError,
                "transaction is complete",
            ):
                journal.append(
                    phase="COMPLETE",
                    owner="NONE",
                    owner_epoch=3,
                    published_tokens_per_request=14,
                    token_history_sha256="d" * 64,
                    phone_active=False,
                    cuda_active=False,
                )

    def test_valid_report_and_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            journal_path = pathlib.Path(directory) / "journal"
            report = self.valid_report(journal_path, flow)
            MOD.validate_report(
                report,
                self.contract,
                self.base,
                journal_path,
            )

    def test_report_from_another_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            journal_path = pathlib.Path(directory) / "journal"
            report = self.valid_report(journal_path, flow)
            with self.assertRaisesRegex(MOD.DeltaError, "run ID mismatch"):
                MOD.validate_report(
                    report,
                    self.contract,
                    self.base,
                    journal_path,
                    "e" * 64,
                )

    def test_run_context_binds_run_and_sources(self):
        VALIDATOR.validate_run_context(
            self.valid_run_context(),
            self.contract,
            self.base,
            self.physical_gate,
        )

    def test_run_context_source_mutation_is_rejected(self):
        context = self.valid_run_context()
        context["sources"]["run_w6_delta_gate.sh"] = "0" * 64
        with self.assertRaisesRegex(MOD.DeltaError, "source digest mismatch"):
            VALIDATOR.validate_run_context(
                context,
                self.contract,
                self.base,
                self.physical_gate,
            )

    def test_realized_placement_logs_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            flow = self.fake_flow()
            report = self.valid_report(root / "journal", flow)
            summary = VALIDATOR.validate_placement_logs(
                self.placement_logs(root, report),
                self.valid_run_context(),
                report,
                self.physical_gate,
            )
            self.assertEqual(set(summary), set(self.physical_gate.placements))

    def test_core_cpu_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            flow = self.fake_flow()
            report = self.valid_report(root / "journal", flow)
            paths = self.placement_logs(
                root,
                report,
                op15={"core_buffer": "CPU"},
            )
            with self.assertRaisesRegex(
                MOD.DeltaError,
                "undeclared compute buffer",
            ):
                VALIDATOR.validate_placement_logs(
                    paths,
                    self.valid_run_context(),
                    report,
                    self.physical_gate,
                )

    def test_executed_step_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            flow = self.fake_flow()
            report = self.valid_report(root / "journal", flow)
            paths = self.placement_logs(
                root,
                report,
                cuda_tail={"step_delta": 1},
            )
            with self.assertRaisesRegex(
                MOD.DeltaError,
                "executed step count mismatch",
            ):
                VALIDATOR.validate_placement_logs(
                    paths,
                    self.valid_run_context(),
                    report,
                    self.physical_gate,
                )

    def test_context_artifact_mutation_is_rejected(self):
        context = self.valid_run_context()
        context["op12"]["worker_sha256"] = "0" * 64
        with self.assertRaisesRegex(MOD.DeltaError, "phone artifact mismatch"):
            VALIDATOR.validate_run_context(
                context,
                self.contract,
                self.base,
                self.physical_gate,
            )

    def test_missing_journal_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            journal_path = pathlib.Path(directory) / "journal"
            report = self.valid_report(journal_path, flow)
            (journal_path / "000003.json").unlink()
            with self.assertRaisesRegex(MOD.DeltaError, "file sequence"):
                MOD.validate_report(
                    report,
                    self.contract,
                    self.base,
                    journal_path,
                )

    def test_delta_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            journal_path = pathlib.Path(directory) / "journal"
            report = self.valid_report(journal_path, flow)
            report["sequences"][0]["phone_delta"][0] += 1
            with self.assertRaisesRegex(MOD.DeltaError, "publication"):
                MOD.validate_report(
                    report,
                    self.contract,
                    self.base,
                    journal_path,
                )

    def test_overlap_accounting_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            journal_path = pathlib.Path(directory) / "journal"
            report = self.valid_report(journal_path, flow)
            report["concurrency"]["overlap_ns"] -= 1
            with self.assertRaisesRegex(MOD.DeltaError, "concurrency accounting"):
                MOD.validate_report(
                    report,
                    self.contract,
                    self.base,
                    journal_path,
                )


if __name__ == "__main__":
    unittest.main()
