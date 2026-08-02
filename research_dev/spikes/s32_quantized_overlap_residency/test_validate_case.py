#!/usr/bin/env python3

import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest


PATH = pathlib.Path(__file__).with_name("validate_case.py")
SPEC = importlib.util.spec_from_file_location("s32_validate_case", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


MODEL_SHA = "a" * 64
REMOTE_MODEL = "/data/local/tmp/model.gguf"


def fixture():
    digest = "b" * 64
    probe = {
        "schema": "s32-quantized-residency-probe-v1",
        "status": "PROBE_PERF_ONLY",
        "scheduler_eligible": False,
        "correctness_status": "NOT_RUN",
        "resource_failures": [],
        "layer_range": [4, 16],
        "quantization": "Q8_0",
        "model_sha256": MODEL_SHA,
        "batch": 32,
        "warmups_discarded": 2,
        "reps": 7,
        "steps": [1],
        "b32_step_us": {"count": 7, "median": 10},
        "b32_per_layer_median_us": 1.0,
        "memory_after": {"process_kib": {"VmSwap": 0, "VmHWM": 100}},
        "measured_cohorts": [
            {"rep": index, "activation_sha256": [digest], "step_us": [10]}
            for index in range(7)
        ],
    }
    compute = {"MUL_MAT": {"HTP0": 10}}
    session = {
        "layer_start": 4,
        "layer_end": 16,
        "missing_buffer_compute_nodes": 0,
        "compute_by_op_and_buffer": compute,
        "session_end": "STOP",
        "placement_status": "SCHEDULED_PLACEMENT_OK",
        "steps_session": 288,
    }
    placement = {
        "layer_start": 4,
        "layer_end": 16,
        "missing_buffer_compute_nodes": 0,
        "compute_by_op_and_buffer": compute,
        "status": "SCHEDULED_PLACEMENT_OK",
        "run_rc": 0,
        "compute_nodes": 10,
        "compute_by_buffer_type": {"HTP0": 10},
    }
    return probe, session, placement


class ValidateCaseTests(unittest.TestCase):
    def run_case(self, mutate=None):
        probe, session, placement = fixture()
        if mutate is not None:
            mutate(probe, session, placement)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            probe_path = root / "probe.json"
            log_path = root / "phone.log"
            hash_path = root / "remote.sha256"
            probe_path.write_bytes(MOD.canonical(probe))
            log_path.write_text(
                "SESSIONCERT " + json.dumps(session) + "\n" +
                "PLACEMENTCERT " + json.dumps(placement) + "\n",
                encoding="ascii",
            )
            hash_path.write_text(f"{MODEL_SHA}  {REMOTE_MODEL}\n", encoding="ascii")
            return MOD.validate_case(
                probe_path, log_path, hash_path, 4, 16, "Q8_0", MODEL_SHA, REMOTE_MODEL,
            )

    def test_valid_perf_only(self):
        result = self.run_case()
        self.assertEqual(result["status"], "CAPACITY_PERF_PASS_NUMERIC_BLOCKED")
        self.assertFalse(result["scheduler_eligible"])

    def test_unrelated_utf8_log_text_is_hashed(self):
        probe, session, placement = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            probe_path = root / "probe.json"
            log_path = root / "phone.log"
            hash_path = root / "remote.sha256"
            probe_path.write_bytes(MOD.canonical(probe))
            log_path.write_bytes(
                "partial load \N{EM DASH} evidence\n".encode("utf-8") +
                b"SESSIONCERT " + json.dumps(session).encode("ascii") + b"\n" +
                b"PLACEMENTCERT " + json.dumps(placement).encode("ascii") + b"\n"
            )
            hash_path.write_text(f"{MODEL_SHA}  {REMOTE_MODEL}\n", encoding="ascii")
            result = MOD.validate_case(
                probe_path, log_path, hash_path, 4, 16, "Q8_0", MODEL_SHA, REMOTE_MODEL,
            )
            self.assertEqual(result["status"], "CAPACITY_PERF_PASS_NUMERIC_BLOCKED")

    def test_rejects_swap(self):
        with self.assertRaisesRegex(MOD.CaseError, "resource failure"):
            self.run_case(lambda probe, _session, _placement: probe["resource_failures"].append("PHONE_SWAP_NONZERO"))

    def test_rejects_cpu_fallback(self):
        def mutate(_probe, session, placement):
            bad = {"MUL_MAT": {"CPU": 10}}
            session["compute_by_op_and_buffer"] = copy.deepcopy(bad)
            placement["compute_by_op_and_buffer"] = copy.deepcopy(bad)
            placement["compute_by_buffer_type"] = {"CPU": 10}
        with self.assertRaisesRegex(MOD.CaseError, "CPU fallback"):
            self.run_case(mutate)

    def test_allows_get_rows_on_cpu(self):
        def mutate(_probe, session, placement):
            good = {"GET_ROWS": {"CPU": 1}, "MUL_MAT": {"HTP0": 10}}
            session["compute_by_op_and_buffer"] = copy.deepcopy(good)
            placement["compute_by_op_and_buffer"] = copy.deepcopy(good)
            placement["compute_nodes"] = 11
            placement["compute_by_buffer_type"] = {"CPU": 1, "HTP0": 10}
        self.assertEqual(self.run_case(mutate)["compute_nodes"], 11)

    def test_rejects_nondeterminism(self):
        def mutate(probe, _session, _placement):
            probe["measured_cohorts"][3]["activation_sha256"] = ["c" * 64]
        with self.assertRaisesRegex(MOD.CaseError, "changed"):
            self.run_case(mutate)

    def test_rejects_eof_session(self):
        with self.assertRaisesRegex(MOD.CaseError, "did not stop"):
            self.run_case(lambda _probe, session, _placement: session.__setitem__("session_end", "EOF"))

    def test_rejects_duplicate_certificate(self):
        with self.assertRaisesRegex(MOD.CaseError, "duplicate JSON key"):
            MOD.decode_json(b'{"run_rc":0,"run_rc":1}', "placement")


if __name__ == "__main__":
    unittest.main()
