#!/usr/bin/env python3

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
SCRIPT = S39 / "warm_tier_controller.py"

SPEC = importlib.util.spec_from_file_location("s39_warm_tier_controller", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONTROLLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLLER)


def load_intents():
    return CONTROLLER.load_intents(
        S39 / "bundle_frequent" / "replay_intents.jsonl",
        S39 / "bundle_frequent" / "replay_manifest.json",
    )


def pass_readiness():
    root = CONTROLLER.load_readiness(S39 / "CURRENT_ROUTE_READINESS.json")
    result = copy.deepcopy(root)
    for route in result.values():
        route["status"] = "PASS"
        route["reason"] = "test fixture"
    return result


class WarmTierControllerTests(unittest.TestCase):
    def test_current_evidence_refuses_trace(self):
        intents = load_intents()
        readiness = CONTROLLER.load_readiness(S39 / "CURRENT_ROUTE_READINESS.json")
        with self.assertRaisesRegex(CONTROLLER.ControllerError, "E_ROUTE_NOT_READY"):
            CONTROLLER.check_trace_ready(intents, readiness)

    def test_cli_refusal_is_nonzero_without_traceback(self):
        process = subprocess.run(
            [sys.executable, str(SCRIPT)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("E_ROUTE_NOT_READY", process.stderr)
        self.assertNotIn("Traceback", process.stderr)
        self.assertEqual(process.stdout, "")

    def test_pass_fixture_accepts_all_nine_intents(self):
        result = CONTROLLER.check_trace_ready(load_intents(), pass_readiness())
        self.assertEqual(result["status"], "READY_FOR_PHYSICAL_REPLAY")
        self.assertEqual(result["intent_count"], 9)

    def test_complete_rotation_follows_frozen_trace(self):
        intents = load_intents()
        readiness = pass_readiness()
        runtime = CONTROLLER.WarmTierController(
            gpu_model=intents[0]["from_model_id"],
            edge_model=intents[0]["to_model_id"],
            readiness=readiness,
        )
        for intent in intents:
            runtime.submit_intent(intent)
            runtime.gpu_drained()
            runtime.gpu_ready()
            runtime.catchup_committed()
            runtime.edge_drained()
            runtime.edge_ready()
        self.assertEqual(runtime.phase, "STABLE")
        self.assertEqual(runtime.gpu_model, intents[-1]["to_model_id"])
        self.assertEqual(runtime.edge_model, intents[-1]["from_model_id"])
        self.assertEqual(len(runtime.actions), 54)
        self.assertEqual(
            [action["kind"] for action in runtime.actions[:6]],
            [
                "DRAIN_GPU",
                "LOAD_GPU",
                "BATCH_CATCHUP",
                "DRAIN_EDGE",
                "PREPARE_EDGE",
                "PUBLISH_EDGE_READY",
            ],
        )

    def test_busy_controller_rejects_next_intent_atomically(self):
        intents = load_intents()
        runtime = CONTROLLER.WarmTierController(
            gpu_model=intents[0]["from_model_id"],
            edge_model=intents[0]["to_model_id"],
            readiness=pass_readiness(),
        )
        runtime.submit_intent(intents[0])
        before = runtime.snapshot()
        with self.assertRaisesRegex(CONTROLLER.ControllerError, "E_TRANSITION_BUSY"):
            runtime.submit_intent(intents[1])
        self.assertEqual(runtime.snapshot(), before)

    def test_nonpass_source_is_rejected_atomically(self):
        intent = load_intents()[0]
        readiness = pass_readiness()
        readiness[intent["from_model_id"]]["status"] = "FAIL_CORRECTNESS"
        runtime = CONTROLLER.WarmTierController(
            gpu_model=intent["from_model_id"],
            edge_model=intent["to_model_id"],
            readiness=readiness,
        )
        before = runtime.snapshot()
        with self.assertRaisesRegex(CONTROLLER.ControllerError, "E_ROUTE_NOT_READY"):
            runtime.submit_intent(intent)
        self.assertEqual(runtime.snapshot(), before)

    def test_nonpass_target_is_rejected_atomically(self):
        intent = load_intents()[0]
        readiness = pass_readiness()
        readiness[intent["to_model_id"]]["status"] = "PROVISIONAL_B1"
        runtime = CONTROLLER.WarmTierController(
            gpu_model=intent["from_model_id"],
            edge_model=intent["to_model_id"],
            readiness=readiness,
        )
        before = runtime.snapshot()
        with self.assertRaisesRegex(CONTROLLER.ControllerError, "E_ROUTE_NOT_READY"):
            runtime.submit_intent(intent)
        self.assertEqual(runtime.snapshot(), before)

    def test_provisional_batch_target_is_rejected_atomically(self):
        intent = load_intents()[0]
        readiness = pass_readiness()
        readiness[intent["to_model_id"]]["status"] = "PROVISIONAL_BATCH"
        runtime = CONTROLLER.WarmTierController(
            gpu_model=intent["from_model_id"],
            edge_model=intent["to_model_id"],
            readiness=readiness,
        )
        before = runtime.snapshot()
        with self.assertRaisesRegex(CONTROLLER.ControllerError, "E_ROUTE_NOT_READY"):
            runtime.submit_intent(intent)
        self.assertEqual(runtime.snapshot(), before)

    def test_phase_order_is_fail_closed(self):
        intent = load_intents()[0]
        runtime = CONTROLLER.WarmTierController(
            gpu_model=intent["from_model_id"],
            edge_model=intent["to_model_id"],
            readiness=pass_readiness(),
        )
        with self.assertRaisesRegex(CONTROLLER.ControllerError, "E_PHASE_GPU_READY"):
            runtime.gpu_ready()
        runtime.submit_intent(intent)
        with self.assertRaisesRegex(CONTROLLER.ControllerError, "E_PHASE_CATCHUP"):
            runtime.catchup_committed()

    def test_intent_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s39_controller_") as directory:
            path = Path(directory) / "replay_intents.jsonl"
            raw = bytearray((S39 / "bundle_frequent" / "replay_intents.jsonl").read_bytes())
            raw[20] ^= 1
            path.write_bytes(raw)
            with self.assertRaisesRegex(CONTROLLER.ControllerError, "SHA-256 mismatch"):
                CONTROLLER.load_intents(
                    path,
                    S39 / "bundle_frequent" / "replay_manifest.json",
                )

    def test_readiness_rejects_float_cut(self):
        value = json.loads((S39 / "CURRENT_ROUTE_READINESS.json").read_text(encoding="ascii"))
        value["routes"]["qwen3-14b-q4_k_m"]["cut_layer"] = 30.0
        with tempfile.TemporaryDirectory(prefix="s39_controller_") as directory:
            path = Path(directory) / "readiness.json"
            path.write_bytes(CONTROLLER.canonical_bytes(value))
            with self.assertRaisesRegex(CONTROLLER.ControllerError, "expected integer"):
                CONTROLLER.load_readiness(path)

    def test_readiness_rejects_duplicate_key(self):
        with tempfile.TemporaryDirectory(prefix="s39_controller_") as directory:
            path = Path(directory) / "readiness.json"
            path.write_text('{"schema_version":1,"schema_version":1}\n', encoding="ascii")
            with self.assertRaisesRegex(CONTROLLER.ControllerError, "duplicate JSON key"):
                CONTROLLER.load_readiness(path)


if __name__ == "__main__":
    unittest.main()
