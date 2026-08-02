#!/usr/bin/env python3

import copy
import importlib.util
import json
import pathlib
import unittest


HERE = pathlib.Path(__file__).resolve().parent
PATH = HERE / "validate_quality.py"
SPEC = importlib.util.spec_from_file_location("s33_validate_quality", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


Q4_DIR = HERE / "results" / "quality_q4_final"
Q8_DIR = HERE / "results" / "quality_q8_v2"
Q4_CORPUS = HERE / "corpus.jsonl"
Q4_MANIFEST = HERE / "corpus_manifest.json"
Q8_CORPUS = HERE / "corpus_q8.jsonl"
Q8_MANIFEST = HERE / "corpus_q8_manifest.json"


def load_probe(path=Q4_DIR / "probe.json"):
    return json.loads(path.read_bytes())


def validate_probe(value, quantization="Q4_0"):
    corpus = Q4_CORPUS if quantization == "Q4_0" else Q8_CORPUS
    manifest = Q4_MANIFEST if quantization == "Q4_0" else Q8_MANIFEST
    return MOD.validate_probe(
        MOD.canonical(value), quantization, corpus.read_bytes(), manifest.read_bytes(),
    )


class ValidateQualityTests(unittest.TestCase):
    def test_real_q4_case_is_an_identity_bound_quality_fail(self):
        result = MOD.validate_case(
            "Q4_0", Q4_DIR, Q4_CORPUS, Q4_MANIFEST,
            "/home/myid/zs89458/Documents/models/gemma-4-12B-it-Q4_0.gguf",
            "/data/local/tmp/ls-s32/gemma-4-12B-it-Q4_0.gguf",
        )
        self.assertEqual(result["status"], "QUALITY_FAIL")
        self.assertTrue(result["resource_identity_bound"])
        self.assertFalse(result["scheduler_eligible"])

    def test_real_q8_case_is_a_quality_fail(self):
        result = MOD.validate_case(
            "Q8_0", Q8_DIR, Q8_CORPUS, Q8_MANIFEST,
            "/home/myid/zs89458/Documents/models/gemma-4-12B-it-Q8_0.gguf",
            "/data/local/tmp/ls-s32/gemma-4-12B-it-Q8_0.gguf",
        )
        self.assertEqual(result["status"], "QUALITY_FAIL")
        self.assertFalse(result["resource_identity_bound"])
        self.assertFalse(result["scheduler_eligible"])

    def test_rejects_token_mutation(self):
        value = load_probe()
        value["physical_tokens"][0][0] += 1
        with self.assertRaises(MOD.ChainError):
            validate_probe(value)

    def test_rejects_false_pass_label(self):
        value = load_probe()
        value["quality_gate_pass"] = True
        with self.assertRaises(MOD.ChainError):
            validate_probe(value)

    def test_rejects_threshold_mutation(self):
        value = load_probe()
        value["thresholds"]["min_first_token_agreement"] = 0.80
        with self.assertRaises(MOD.ChainError):
            validate_probe(value)

    def test_rejects_memory_pid_mutation(self):
        value = load_probe()
        value["memory"]["head_after"]["pid"] += 1
        with self.assertRaises(MOD.ChainError):
            validate_probe(value)

    def test_rejects_model_hash_mutation(self):
        raw = (Q4_DIR / "host_model.sha256").read_bytes()
        with self.assertRaises(MOD.ChainError):
            MOD.parse_hash_record(
                b"0" * 64 + raw[64:],
                MOD.CONFIGS["Q4_0"]["model_sha256"],
                "/home/myid/zs89458/Documents/models/gemma-4-12B-it-Q4_0.gguf",
                "host",
            )

    def test_q4_and_q8_corpus_tokens_are_identical(self):
        self.assertEqual(Q4_CORPUS.read_bytes(), Q8_CORPUS.read_bytes())

    def test_pass_cannot_bind_without_memory_identity(self):
        value = load_probe()
        value["physical_tokens"] = copy.deepcopy(value["reference_tokens"])
        value["physical_tokens_sha256"] = MOD.sha256(MOD.canonical(value["physical_tokens"]))
        value["quality"] = {
            "exact_sequence_agreement": 1.0,
            "exact_sequence_matches": 128,
            "first_token_agreement": 1.0,
            "first_token_matches": 128,
            "token_decision_agreement": 1.0,
            "token_decision_matches": 1024,
        }
        value["quality_gate_pass"] = True
        value["status"] = "QUALITY_PASS"
        for snapshot in value["memory"].values():
            snapshot.pop("adb_serial")
            snapshot.pop("pid")
            snapshot["process_kib"].pop("Pid")
        _, quality_pass, identity_bound = validate_probe(value)
        self.assertTrue(quality_pass)
        self.assertFalse(identity_bound)

    def test_rejects_compute_fallback(self):
        raw = (Q4_DIR / "op15.log").read_bytes()
        mutated = raw.replace(b'"MUL_MAT":{"HTP0":', b'"MUL_MAT":{"CPU":')
        with self.assertRaises(MOD.ChainError):
            MOD.validate_worker(mutated, None, 4, 24, "HTP0", 1920, "OP15")


if __name__ == "__main__":
    unittest.main()
