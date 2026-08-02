#!/usr/bin/env python3

import importlib.util
import json
import pathlib
import tempfile
import unittest


PATH = pathlib.Path(__file__).with_name("quant_quality_probe.py")
SPEC = importlib.util.spec_from_file_location("s33_quality", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def hello(start, end, terminal=False, streams=32):
    return MOD.Hello(
        start, end, 48, 3840, streams, 16, 256, 256,
        0x0F | (0x10 if terminal else 0),
    )


class QualityProbeTests(unittest.TestCase):
    def test_topology(self):
        MOD.validate_topology(
            hello(0, 4), hello(4, 24), hello(24, 48, True), hello(0, 24),
            4, 24,
        )

    def test_q8_topology(self):
        MOD.validate_topology(
            hello(0, 2), hello(2, 16), hello(16, 48, True), hello(0, 16),
            2, 16,
        )

    def test_topology_rejects_gap(self):
        with self.assertRaises(MOD.ProtocolError):
            MOD.validate_topology(
                hello(0, 4), hello(5, 24), hello(24, 48, True), hello(0, 24),
                4, 24,
            )

    def test_topology_rejects_small_batch(self):
        with self.assertRaises(MOD.ProtocolError):
            MOD.validate_topology(
                hello(0, 4, streams=31), hello(4, 24),
                hello(24, 48, True), hello(0, 24),
                4, 24,
            )

    def test_quality_summary(self):
        physical = [[1] * 8 for _ in range(128)]
        reference = [[1] * 8 for _ in range(128)]
        reference[0][-1] = 2
        result = MOD.quality_summary(physical, reference)
        self.assertEqual(result["first_token_matches"], 128)
        self.assertEqual(result["token_decision_matches"], 1023)
        self.assertEqual(result["exact_sequence_matches"], 127)

    def test_hidden_rows_reject_stale_lineage(self):
        result = MOD.BatchResult(8, 9, 0, 0, (1.0,) * 3840, None)
        with self.assertRaises(MOD.ProtocolError):
            MOD.hidden_rows([result] * 32, [2] * 32, 0, 100)

    def test_memory_failures(self):
        memory = {
            f"{device}_{phase}": {
                "adb_serial": device,
                "pid": 10 if device == "head" else 20,
                "process_kib": {"VmSwap": 0},
            }
            for device in ("head", "middle")
            for phase in ("before", "after")
        }
        self.assertEqual(MOD.memory_failures(memory), [])
        memory["middle_after"]["process_kib"]["VmSwap"] = 4
        self.assertEqual(
            MOD.memory_failures(memory), ["MIDDLE_AFTER_SWAP_NONZERO"],
        )

    def test_parse_kib_fields(self):
        self.assertEqual(
            MOD.parse_kib_fields("Pid:\t7\nVmRSS:\t123 kB\nVmSwap:\t0 kB\n"),
            {"Pid": 7, "VmRSS": 123, "VmSwap": 0},
        )

    def test_memory_rejects_identity_change(self):
        memory = {
            f"{device}_{phase}": {
                "adb_serial": device,
                "pid": 10,
                "process_kib": {"VmSwap": 0},
            }
            for device in ("head", "middle")
            for phase in ("before", "after")
        }
        memory["head_after"]["pid"] = 11
        with self.assertRaises(ValueError):
            MOD.memory_failures(memory)

    def test_load_corpus(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "corpus.jsonl"
            rows = []
            for prompt_id in range(128):
                rows.append(MOD.canonical({
                    "schema": MOD.CORPUS_SCHEMA,
                    "prompt_id": prompt_id,
                    "source_row": prompt_id,
                    "text_sha256": f"{prompt_id:064x}",
                    "source_token_count": 8,
                    "tokens": list(range(8)),
                }))
            path.write_bytes(b"".join(rows))
            records, digest = MOD.load_corpus(path)
            self.assertEqual(len(records), 128)
            self.assertEqual(digest, MOD.sha256(path.read_bytes()))

    def test_load_corpus_rejects_duplicate_key(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "corpus.jsonl"
            path.write_text('{"schema":"x","schema":"x"}\n', encoding="ascii")
            with self.assertRaises(ValueError):
                MOD.load_corpus(path)


if __name__ == "__main__":
    unittest.main()
