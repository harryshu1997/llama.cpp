#!/usr/bin/env python3

import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).parents[1]
PATH = ROOT / "validate_qwen25_quality.py"
SPEC = importlib.util.spec_from_file_location("validate_qwen25_quality", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)

EVIDENCE = ROOT / "results" / "w4_qwen25_route" / "gpu_quality"
CORPUS = ROOT / "results" / "w4_qwen25_route" / "corpus_qwen25_q4_0.jsonl"
MANIFEST = (
    ROOT
    / "results"
    / "w4_qwen25_route"
    / "corpus_qwen25_q4_0_manifest.json"
)
Q8_EVIDENCE = ROOT / "results" / "w4_qwen25_route" / "q8_quality_final"
Q8_CORPUS = ROOT / "results" / "w4_qwen25_route" / "corpus_qwen25_q8_0.jsonl"
Q8_MANIFEST = (
    ROOT
    / "results"
    / "w4_qwen25_route"
    / "corpus_qwen25_q8_0_manifest.json"
)
Q8_SPEC = MOD.RouteSpec(
    model_sha256="23ca481b8226b2492ba8f3eb7af41e0f99d8605c16fb6dec7bc5cf6716b673cf",
    file_type=7,
    cut_layer=30,
    op15_shard_sha256="b9611440eb4901764cef6afb55e08418acb114f1dfdc5b97e2c4acafefcc5375",
    op12_shard_sha256="b66f9f6ace28da341f21f4f0d03ffa05c31cea647d43da4023e373f6021551ac",
    op15_adb_target="3C15AU002CL00000",
    op12_adb_target="5ae7a43d",
)


class Qwen25QualityValidatorTests(unittest.TestCase):
    def test_q8_route_spec_binds_hello_identity(self):
        spec = MOD.RouteSpec(
            model_sha256="1" * 64,
            file_type=7,
            cut_layer=30,
            op15_shard_sha256="2" * 64,
            op12_shard_sha256="3" * 64,
            op15_adb_target="op15",
            op12_adb_target="op12",
        )
        hello = {
            "file_type": 7,
            "layer_end": MOD.N_LAYER,
            "layer_start": 0,
            "max_streams": MOD.BATCH,
            "model_sha256": "1" * 64,
            "n_batch": MOD.BATCH * MOD.PREFILL_CHUNK,
            "n_ctx_seq": MOD.PROMPT_TOKENS + MOD.OUTPUT_TOKENS - 1,
            "n_embd": MOD.N_EMBD,
            "n_layer": MOD.N_LAYER,
            "n_ubatch": MOD.BATCH * MOD.PREFILL_CHUNK,
        }
        MOD.validate_hello(hello, "hello", spec)
        hello["file_type"] = 2
        with self.assertRaisesRegex(MOD.EvidenceError, "file_type"):
            MOD.validate_hello(hello, "hello", spec)

    def test_route_spec_rejects_unknown_file_type(self):
        spec = MOD.DEFAULT_SPEC._replace(file_type=99)
        with self.assertRaisesRegex(MOD.EvidenceError, "route_spec.file_type"):
            MOD.validate_spec(spec)

    def validate(self, root: pathlib.Path):
        return MOD.validate_report(
            root / "quality_gpu_report.json",
            CORPUS,
            MANIFEST,
            root,
        )

    def copy_evidence(self, root: pathlib.Path):
        for source in EVIDENCE.iterdir():
            if source.is_file():
                (root / source.name).write_bytes(source.read_bytes())

    def mutate_json(self, path: pathlib.Path, update):
        value = json.loads(path.read_text(encoding="ascii"))
        update(value)
        path.write_bytes(MOD.canonical(value))

    def test_real_evidence_derives_quality_fail(self):
        certificate = self.validate(EVIDENCE)
        self.assertEqual(certificate["status"], "QUALITY_FAIL")
        self.assertFalse(certificate["scheduler_eligible"])
        self.assertEqual(
            certificate["quality"]["token_decision_matches"],
            880,
        )

    def test_real_q8_evidence_derives_quality_fail(self):
        certificate = MOD.validate_report(
            Q8_EVIDENCE / "quality_q8_report.json",
            Q8_CORPUS,
            Q8_MANIFEST,
            Q8_EVIDENCE,
            Q8_SPEC,
        )
        self.assertEqual(certificate["status"], "QUALITY_FAIL")
        self.assertFalse(certificate["scheduler_eligible"])
        self.assertEqual(
            certificate["quality"]["token_decision_matches"],
            872,
        )

    def test_token_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.copy_evidence(root)
            self.mutate_json(
                root / "quality_gpu_report.json",
                lambda value: value["physical_tokens"][0].__setitem__(
                    0,
                    value["physical_tokens"][0][0] + 1,
                ),
            )
            with self.assertRaisesRegex(
                MOD.EvidenceError,
                "physical_tokens_sha256",
            ):
                self.validate(root)

    def test_quality_label_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.copy_evidence(root)
            self.mutate_json(
                root / "quality_gpu_report.json",
                lambda value: value.__setitem__("quality_gate_pass", True),
            )
            with self.assertRaisesRegex(
                MOD.EvidenceError,
                "quality_gate_pass",
            ):
                self.validate(root)

    def test_relay_row_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.copy_evidence(root)
            path = root / "phone_relay.log"
            text = path.read_text(encoding="utf-8")
            text = text.replace('"rows":1920', '"rows":1919')
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(MOD.EvidenceError, "rows"):
                self.validate(root)

    def test_undeclared_cpu_compute_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.copy_evidence(root)
            path = root / "op15_head.log"
            text = path.read_text(encoding="utf-8")
            text = text.replace(
                '"MUL_MAT":{"OpenCL":9856}',
                '"MUL_MAT":{"CPU":1,"OpenCL":9856}',
            )
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(MOD.EvidenceError, "undeclared host"):
                self.validate(root)

    def test_post_run_boot_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.copy_evidence(root)
            self.mutate_json(
                root / "POST_RUN_CONTEXT.json",
                lambda value: value["op15"].__setitem__("boot_id", "wrong"),
            )
            with self.assertRaisesRegex(MOD.EvidenceError, "op15.boot_id"):
                self.validate(root)

    def test_duplicate_report_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.copy_evidence(root)
            path = root / "quality_gpu_report.json"
            raw = path.read_bytes()
            path.write_bytes(raw.replace(b'{"batch":32', b'{"batch":32,"batch":32'))
            with self.assertRaisesRegex(MOD.EvidenceError, "duplicate JSON key"):
                self.validate(root)


if __name__ == "__main__":
    unittest.main()
