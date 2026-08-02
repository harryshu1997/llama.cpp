#!/usr/bin/env python3

import importlib.util
import json
import pathlib
import tempfile
import unittest


PATH = pathlib.Path(__file__).parents[1] / "qwen25_quality_probe.py"
SPEC = importlib.util.spec_from_file_location("qwen25_quality_probe", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class Qwen25QualityProbeTests(unittest.TestCase):
    def test_phase_call_labels_socket_error(self):
        def fail():
            raise ConnectionRefusedError(111, "Connection refused")

        with self.assertRaisesRegex(
            MOD.ProtocolError,
            r"CONNECT_PHONE: \[Errno 111\] Connection refused",
        ):
            MOD.phase_call("CONNECT_PHONE", fail)

    def test_quality_summary_exact(self):
        values = [[index] * MOD.OUTPUT_TOKENS for index in range(MOD.PROMPTS)]
        summary = MOD.quality_summary(values, values)
        self.assertEqual(summary["first_token_agreement"], 1.0)
        self.assertEqual(summary["token_decision_agreement"], 1.0)
        self.assertEqual(summary["exact_sequence_agreement"], 1.0)

    def test_quality_summary_counts_divergence(self):
        reference = [[index] * MOD.OUTPUT_TOKENS for index in range(MOD.PROMPTS)]
        physical = [list(row) for row in reference]
        physical[0][0] += 1
        summary = MOD.quality_summary(physical, reference)
        self.assertEqual(summary["first_token_matches"], MOD.PROMPTS - 1)
        self.assertEqual(
            summary["token_decision_matches"],
            MOD.PROMPTS * MOD.OUTPUT_TOKENS - 1,
        )
        self.assertEqual(summary["exact_sequence_matches"], MOD.PROMPTS - 1)

    def test_prefill_rows_are_sequence_major(self):
        prompts = [
            [seq_id * 100 + position for position in range(MOD.PROMPT_TOKENS)]
            for seq_id in range(MOD.BATCH)
        ]
        rows = MOD.prefill_rows(prompts, 1000, 2, 4)
        self.assertEqual(len(rows), MOD.BATCH * MOD.PREFILL_CHUNK)
        self.assertEqual(
            [(row.seq_id, row.position, row.token) for row in rows[:10]],
            [
                (0, 2, 2), (0, 3, 3),
                (1, 2, 102), (1, 3, 103),
                (2, 2, 202), (2, 3, 203),
                (3, 2, 302), (3, 3, 303),
                (4, 2, 402), (4, 3, 403),
            ],
        )

    def test_prefill_rows_reject_large_chunk(self):
        prompts = [list(range(MOD.PROMPT_TOKENS)) for _ in range(MOD.BATCH)]
        with self.assertRaisesRegex(MOD.ProtocolError, "prefill chunk"):
            MOD.prefill_rows(prompts, 1000, 0, MOD.PREFILL_CHUNK + 1)

    def test_load_corpus_rejects_model_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            corpus = root / "corpus.jsonl"
            manifest = root / "manifest.json"
            records = [
                {
                    "prompt_id": index,
                    "schema": MOD.CORPUS_SCHEMA,
                    "source_row": index,
                    "source_token_count": MOD.PROMPT_TOKENS,
                    "text_sha256": f"{index:064x}",
                    "tokens": list(range(MOD.PROMPT_TOKENS)),
                }
                for index in range(MOD.PROMPTS)
            ]
            corpus_raw = b"".join(MOD.canonical(record) for record in records)
            corpus.write_bytes(corpus_raw)
            value = {
                "model_sha256": "0" * 64,
                "output_records": MOD.PROMPTS,
                "output_sha256": MOD.sha256(corpus_raw),
                "schema": MOD.MANIFEST_SCHEMA,
                "selection": {"prompt_tokens": MOD.PROMPT_TOKENS},
                "source_revision": MOD.SOURCE_REVISION,
                "source_sha256": MOD.SOURCE_SHA256,
            }
            manifest.write_bytes(MOD.canonical(value))
            with self.assertRaisesRegex(ValueError, "manifest binding"):
                MOD.load_corpus(corpus, manifest)

    def test_load_corpus_accepts_explicit_model_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            corpus = root / "corpus.jsonl"
            manifest = root / "manifest.json"
            records = [
                {
                    "prompt_id": index,
                    "schema": MOD.CORPUS_SCHEMA,
                    "source_row": index,
                    "source_token_count": MOD.PROMPT_TOKENS,
                    "text_sha256": f"{index:064x}",
                    "tokens": list(range(MOD.PROMPT_TOKENS)),
                }
                for index in range(MOD.PROMPTS)
            ]
            corpus_raw = b"".join(MOD.canonical(record) for record in records)
            corpus.write_bytes(corpus_raw)
            model_sha256 = "1" * 64
            value = {
                "model_sha256": model_sha256,
                "output_records": MOD.PROMPTS,
                "output_sha256": MOD.sha256(corpus_raw),
                "schema": MOD.MANIFEST_SCHEMA,
                "selection": {"prompt_tokens": MOD.PROMPT_TOKENS},
                "source_revision": MOD.SOURCE_REVISION,
                "source_sha256": MOD.SOURCE_SHA256,
            }
            manifest.write_bytes(MOD.canonical(value))
            loaded, corpus_digest, _ = MOD.load_corpus(
                corpus,
                manifest,
                model_sha256,
            )
            self.assertEqual(loaded, records)
            self.assertEqual(corpus_digest, MOD.sha256(corpus_raw))

    def test_strict_json_rejects_duplicate_key(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            json.loads('{"a":1,"a":2}', object_pairs_hook=MOD.strict_object)


if __name__ == "__main__":
    unittest.main()
