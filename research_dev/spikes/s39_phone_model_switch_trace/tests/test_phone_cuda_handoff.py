#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace


PATH = pathlib.Path(__file__).parents[1] / "phone_cuda_handoff_probe.py"
SPEC = importlib.util.spec_from_file_location("phone_cuda_handoff_probe", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)

VALIDATOR_PATH = pathlib.Path(__file__).parents[1] / "validate_phone_cuda_handoff.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_phone_cuda_handoff",
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
        return (sum((index + 1) * token for index, token in enumerate(history)) + 17) % 50000

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


class PhoneCudaHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = MOD.load_contract(MOD.DEFAULT_CONTRACT)

    def histories(self):
        return [
            [seq_id * 100 + position for position in range(12)]
            for seq_id in range(self.contract.batch)
        ]

    def test_frozen_contract_loads(self):
        contract = self.contract
        self.assertEqual(contract.batch, 8)
        self.assertEqual(contract.cuda_catchup_chunk, 8)
        self.assertFalse(
            MOD.json.loads(MOD.DEFAULT_CONTRACT.read_bytes())[
                "scheduler_eligible_on_pass"
            ]
        )

    def test_replay_is_exact_across_chunk_sizes(self):
        histories = self.histories()
        control, control_metrics = MOD.run_replay(
            FakeClient(), histories, 1000, 2, 8,
        )
        catchup, catchup_metrics = MOD.run_replay(
            FakeClient(), histories, 2000, 8, 8,
        )
        self.assertEqual(catchup, control)
        self.assertLess(
            catchup_metrics.history_batches,
            control_metrics.history_batches,
        )

    def test_remove_group_releases_every_sequence(self):
        client = FakeClient()
        MOD.run_replay(client, self.histories(), 1000, 8, 1)
        MOD.remove_group(client, self.contract.batch, 1000)
        self.assertEqual(client.histories, {})

    def valid_report(self):
        contract = self.contract
        prompts = [
            [seq_id * 10 + position for position in range(contract.prompt_tokens)]
            for seq_id in range(contract.batch)
        ]
        phone = [
            [40000 + seq_id * 10 + index for index in range(contract.phone_committed_tokens)]
            for seq_id in range(contract.batch)
        ]
        continuation = [
            [45000 + seq_id * 10 + index for index in range(contract.cuda_continuation_tokens)]
            for seq_id in range(contract.batch)
        ]
        histories = [
            prompt + committed
            for prompt, committed in zip(prompts, phone)
        ]
        history_sha = MOD.histories_digest(histories)
        sequences = [
            {
                "catchup_continuation": continuation[index],
                "committed_history": histories[index],
                "control_continuation": list(continuation[index]),
                "phone_committed": phone[index],
                "prompt_id": contract.prompt_ids[index],
                "prompt_tokens": prompts[index],
                "published_tokens": phone[index] + continuation[index],
                "sequence_index": index,
            }
            for index in range(contract.batch)
        ]
        metrics = {
            "continuation_batches": 7,
            "elapsed_us": 100,
            "history_batches": 6,
            "history_sha256": history_sha,
            "rows": 152,
        }
        catchup = dict(metrics)
        catchup["history_batches"] = 2
        hello = {
            "capabilities": (
                MOD.STAGE_V3_CAP_TERMINAL
                | 0x0F
                | 0x20
            ),
            "file_type": contract.file_type,
            "layer_end": contract.n_layer,
            "layer_start": 0,
            "max_streams": contract.batch,
            "model_sha256": contract.model_sha256,
            "n_batch": contract.max_rows_per_batch,
            "n_ctx_seq": 256,
            "n_embd": contract.n_embd,
            "n_layer": contract.n_layer,
            "n_ubatch": contract.max_rows_per_batch,
        }
        return {
            "batch": contract.batch,
            "contract_sha256": contract.raw_sha256,
            "corpus_manifest_sha256": contract.manifest_sha256,
            "corpus_sha256": contract.corpus_sha256,
            "cuda_catchup": catchup,
            "cuda_continuation_tokens": contract.cuda_continuation_tokens,
            "cuda_control": metrics,
            "events": MOD.expected_events(
                contract.phone_committed_tokens,
                contract.cuda_continuation_tokens,
            ),
            "hellos": {
                "cuda": dict(hello),
                "phone": dict(hello),
            },
            "model_sha256": contract.model_sha256,
            "phone_committed_tokens": contract.phone_committed_tokens,
            "prompts": list(contract.prompt_ids),
            "scheduler_eligible": False,
            "schema": MOD.SCHEMA,
            "scope": "MECHANICS_ONLY",
            "sequences": sequences,
            "state_counts": {
                "cuda_after_completion": 0,
                "cuda_control_released": 0,
                "cuda_prepared": contract.batch,
                "phone_frontier": contract.batch,
                "phone_released": 0,
            },
            "status": "HANDOFF_MECHANICS_PASS",
        }

    def test_report_validator_accepts_exact_handoff(self):
        MOD.validate_report(self.valid_report(), self.contract)

    def test_report_validator_rejects_missing_committed_token(self):
        report = self.valid_report()
        report["sequences"][0]["committed_history"].pop()
        with self.assertRaisesRegex(MOD.HandoffError, "history has a gap"):
            MOD.validate_report(report, self.contract)

    def test_report_validator_rejects_duplicate_published_token(self):
        report = self.valid_report()
        report["sequences"][0]["published_tokens"].insert(0, 1)
        with self.assertRaisesRegex(MOD.HandoffError, "gap or duplicate"):
            MOD.validate_report(report, self.contract)

    def test_report_validator_rejects_continuation_mismatch(self):
        report = self.valid_report()
        report["sequences"][0]["catchup_continuation"][0] += 1
        report["sequences"][0]["published_tokens"][4] += 1
        with self.assertRaisesRegex(MOD.HandoffError, "status mismatch"):
            MOD.validate_report(report, self.contract)

    def test_report_validator_rejects_false_batch_claim(self):
        report = self.valid_report()
        report["cuda_catchup"]["history_batches"] = 6
        with self.assertRaisesRegex(MOD.HandoffError, "replay accounting"):
            MOD.validate_report(report, self.contract)

    def test_report_validator_rejects_scheduler_eligibility(self):
        report = self.valid_report()
        report["scheduler_eligible"] = True
        with self.assertRaisesRegex(MOD.HandoffError, "eligibility"):
            MOD.validate_report(report, self.contract)

    def test_report_validator_rejects_owner_epoch_mutation(self):
        report = self.valid_report()
        report["events"][3]["owner_epoch"] = 1
        with self.assertRaisesRegex(MOD.HandoffError, "ownership event"):
            MOD.validate_report(report, self.contract)

    def test_report_validator_rejects_history_digest_mutation(self):
        report = self.valid_report()
        report["cuda_control"]["history_sha256"] = "0" * 64
        with self.assertRaisesRegex(MOD.HandoffError, "history digest"):
            MOD.validate_report(report, self.contract)

    def test_report_validator_rejects_false_state_count(self):
        report = self.valid_report()
        report["state_counts"]["phone_released"] = 1
        with self.assertRaisesRegex(MOD.HandoffError, "sequence-state"):
            MOD.validate_report(report, self.contract)

    def test_report_validator_rejects_false_row_count(self):
        report = self.valid_report()
        report["cuda_catchup"]["rows"] -= 1
        with self.assertRaisesRegex(MOD.HandoffError, "replay accounting"):
            MOD.validate_report(report, self.contract)

    def test_report_validator_rejects_hello_model_mutation(self):
        report = self.valid_report()
        report["hellos"]["phone"]["model_sha256"] = "0" * 64
        with self.assertRaisesRegex(MOD.HandoffError, "model SHA-256 mismatch"):
            MOD.validate_report(report, self.contract)

    def valid_run_context(self):
        contract = self.contract
        sources = {
            "async_pipeline.py": (
                VALIDATOR.HERE.parent
                / "s22_slo_overlap_pipeline"
                / "async_pipeline.py"
            ),
            "phone_cuda_handoff_probe.py": (
                VALIDATOR.HERE / "phone_cuda_handoff_probe.py"
            ),
            "qwen25_quality_probe.py": (
                VALIDATOR.HERE / "qwen25_quality_probe.py"
            ),
            "run_w5_handoff_gate.sh": (
                VALIDATOR.HERE / "run_w5_handoff_gate.sh"
            ),
            "stage_v3_client.py": (
                VALIDATOR.HERE.parent
                / "s22_slo_overlap_pipeline"
                / "stage_v3_client.py"
            ),
            "validate_phone_cuda_handoff.py": (
                VALIDATOR.HERE / "validate_phone_cuda_handoff.py"
            ),
        }
        return {
            "acquisition_unix_s": 1,
            "base_git_commit": "0" * 40,
            "contract_sha256": contract.raw_sha256,
            "cuda": {
                "device": "CUDA0",
                "head_port": 41310,
                "relay_port": 41312,
                "relay_sha256": "1" * 64,
                "tail_port": 41311,
                "worker_sha256": "2" * 64,
            },
            "model_sha256": contract.model_sha256,
            "op12": {
                "adb_target": "op12",
                "boot_id": "boot12",
                "layers": [30, 48],
                "shard_sha256": "3" * 64,
                "wifi": "127.0.0.12",
                "worker_sha256": "4" * 64,
            },
            "op15": {
                "adb_target": "op15",
                "boot_id": "boot15",
                "layers": [0, 30],
                "relay_sha256": "5" * 64,
                "shard_sha256": "6" * 64,
                "wifi": "127.0.0.15",
                "worker_sha256": "7" * 64,
            },
            "schema": "s39-phone-cuda-handoff-context-v1",
            "sources": {
                name: MOD.sha256(path.read_bytes())
                for name, path in sources.items()
            },
        }

    def test_run_context_binds_all_sources(self):
        VALIDATOR.validate_run_context(
            self.valid_run_context(),
            self.contract,
        )

    def test_run_context_rejects_source_digest_mutation(self):
        context = self.valid_run_context()
        context["sources"]["stage_v3_client.py"] = "0" * 64
        with self.assertRaisesRegex(MOD.HandoffError, "source digest mismatch"):
            VALIDATOR.validate_run_context(context, self.contract)


if __name__ == "__main__":
    unittest.main()
