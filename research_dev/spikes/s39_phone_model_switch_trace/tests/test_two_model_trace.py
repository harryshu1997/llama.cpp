#!/usr/bin/env python3

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
SCRIPT = S39 / "validate_two_model_trace.py"

SPEC = importlib.util.spec_from_file_location("s39_two_model_trace", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


class TwoModelTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = TRACE.build_contract(S39)

    def test_frozen_trace_requires_both_models_on_both_tiers(self):
        self.assertEqual(
            self.contract["status"],
            "TRACE_CONTRACT_PASS_PHYSICAL_EXECUTION_NOT_RUN",
        )
        self.assertEqual(self.contract["successful_requests"], 74)
        summaries = self.contract["model_summary"]
        self.assertEqual(
            summaries["gemma-4-12b-it-q4_0"]["successful_requests"],
            57,
        )
        self.assertEqual(
            summaries["qwen3-14b-q4_k_m"]["successful_requests"],
            17,
        )
        for summary in summaries.values():
            self.assertTrue(summary["phone_trigger_event_ids"])
            self.assertTrue(summary["cuda_witness_event_ids"])

    def test_trace_has_nine_bidirectional_transitions(self):
        transitions = self.contract["transitions"]
        self.assertEqual(len(transitions), 9)
        self.assertEqual(
            [transition["kind"] for transition in transitions],
            ["PROMOTE", "DEMOTE"] * 4 + ["PROMOTE"],
        )
        self.assertEqual(
            self.contract["initial_planned_state"],
            {
                "gpu_model_id": "gemma-4-12b-it-q4_0",
                "phone_model_id": "qwen3-14b-q4_k_m",
            },
        )
        self.assertEqual(
            self.contract["final_planned_state"],
            {
                "gpu_model_id": "qwen3-14b-q4_k_m",
                "phone_model_id": "gemma-4-12b-it-q4_0",
            },
        )

    def test_trigger_model_mismatch_is_rejected(self):
        transition = copy.deepcopy(self.contract["transitions"])
        self.assertTrue(transition)
        intents = [
            json.loads(line)
            for line in (S39 / "bundle_frequent" / "replay_intents.jsonl")
            .read_text(encoding="ascii")
            .splitlines()
        ]
        requests = [
            json.loads(line)
            for line in (S39 / "bundle_frequent" / "requests.jsonl")
            .read_text(encoding="ascii")
            .splitlines()
        ]
        assignment = json.loads(
            (S39 / "bundle_frequent" / "model_assignment.json").read_text(
                encoding="ascii"
            )
        )
        trigger_id = intents[0]["source_event_id"]
        for request in requests:
            if request["event_id"] == trigger_id:
                request["source_fields"]["model"] = "ChatGPT"
                break
        with self.assertRaisesRegex(TRACE.TraceContractError, "E_TRIGGER_MODEL"):
            TRACE.derive_contract(
                requests=requests,
                mappings=assignment["mappings"],
                intents=intents,
                hot_model="gemma-4-12b-it-q4_0",
                cold_model="qwen3-14b-q4_k_m",
                horizon_us=1_200_000_000,
                input_digests={},
            )

    def test_missing_second_model_requests_is_rejected(self):
        intents = [
            json.loads(line)
            for line in (S39 / "bundle_frequent" / "replay_intents.jsonl")
            .read_text(encoding="ascii")
            .splitlines()
        ]
        requests = [
            json.loads(line)
            for line in (S39 / "bundle_frequent" / "requests.jsonl")
            .read_text(encoding="ascii")
            .splitlines()
        ]
        assignment = json.loads(
            (S39 / "bundle_frequent" / "model_assignment.json").read_text(
                encoding="ascii"
            )
        )
        requests = [
            request
            for request in requests
            if request["source_fields"]["model"] != "ChatGPT"
        ]
        with self.assertRaisesRegex(TRACE.TraceContractError, "E_TWO_MODEL_REQUESTS"):
            TRACE.derive_contract(
                requests=requests,
                mappings=assignment["mappings"],
                intents=intents,
                hot_model="gemma-4-12b-it-q4_0",
                cold_model="qwen3-14b-q4_k_m",
                horizon_us=1_200_000_000,
                input_digests={},
            )

    def test_cli_is_deterministic_across_hash_seeds(self):
        outputs = []
        with tempfile.TemporaryDirectory(prefix="s39_two_model_cli_") as directory:
            for seed in ("0", "1", "42", "12345"):
                output = Path(directory) / f"contract-{seed}.json"
                process = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--root",
                        str(S39),
                        "--output",
                        str(output),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONHASHSEED": seed,
                    },
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                outputs.append(output.read_bytes())
        self.assertTrue(all(raw == outputs[0] for raw in outputs[1:]))


if __name__ == "__main__":
    unittest.main()
