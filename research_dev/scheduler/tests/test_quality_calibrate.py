"""Off-rig quality calibration (quality/calibrate.py): item plan, sizing rule, request body, token offsets, run
bookkeeping and the paired report, on synthetic items and fabricated runs (no GPU, no model files)."""
import argparse
import json
import random
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

from research_dev.scheduler.adapters import http_backend
from research_dev.scheduler.campaigns.burstgpt.quality import calibrate, protocol
from research_dev.scheduler.campaigns.burstgpt.quality.gsm8k import (
    EXPECTED_FIRST_TOKENS, EXPECTED_LAST_TOKENS, Gsm8kItem,
)
from research_dev.scheduler.campaigns.burstgpt.quality.stats import PairedCounts, newcombe_interval, tango_interval

SPLIT_ROWS = 400


def train_items(rows=SPLIT_ROWS):
    return [Gsm8kItem(split="train", index=index, question=f"Question {index}?", reference=Decimal(index % 50))
            for index in range(rows)]


def response(tokens, content, predicted=None):
    return json.dumps({"content": content, "tokens": tokens, "stop": True,
                       "tokens_predicted": len(tokens) if predicted is None else predicted}).encode("ascii")


def answer_output(value, filler=7):
    """A full-budget output whose earliest conclusion is `value`."""
    return list(range(calibrate.OUTPUT_TOKENS)), f"Reasoning.\nFinal answer: {value}\nmore text " + "x" * filler


class ItemPlanTests(unittest.TestCase):
    def test_pilot_exclusion_is_the_pilot_builders_first_items_of_its_permutation(self):
        items = train_items()
        order = list(range(SPLIT_ROWS))
        random.Random(protocol.PILOT_SEED).shuffle(order)  # independent restatement of build_trace --pilot
        expected = set(order[:3 * protocol.PILOT_PER_MODEL])
        self.assertEqual(calibrate.pilot_item_indices(items), expected)

    def test_sequences_are_disjoint_deterministic_and_exclude_the_pilot(self):
        items = train_items()
        first, second = calibrate.calibration_sequences(items), calibrate.calibration_sequences(items)
        self.assertEqual(first, second)
        qwen = {item.index for item in first["qwen"]}
        gemma = {item.index for item in first["gemma"]}
        pilot = calibrate.pilot_item_indices(items)
        self.assertFalse(qwen & gemma)
        self.assertFalse((qwen | gemma) & pilot)
        self.assertEqual(qwen | gemma | pilot, set(range(SPLIT_ROWS)))
        order = [item.index for item in items if item.index not in pilot]
        random.Random(calibrate.CALIBRATION_SEED).shuffle(order)
        self.assertEqual([item.index for item in first["qwen"]], order[0::2])
        self.assertEqual([item.index for item in first["gemma"]], order[1::2])

    def test_test_split_items_are_refused(self):
        items = train_items()
        items[3] = Gsm8kItem(split="test", index=3, question="q", reference=Decimal(1))
        with self.assertRaises(calibrate.CalibrationError):
            calibrate.calibration_sequences(items)


class SizingRuleTests(unittest.TestCase):
    def test_rule_values(self):
        # n = 1.96^2 psi / 0.015^2, rounded up to 100, clamped to [200, 3000]
        self.assertEqual(calibrate.required_items(0.0), 200)
        self.assertEqual(calibrate.required_items(0.02), 400)    # 341.5
        self.assertEqual(calibrate.required_items(0.05), 900)    # 853.7
        self.assertEqual(calibrate.required_items(0.10), 1800)   # 1707.3
        self.assertEqual(calibrate.required_items(0.30), 3000)   # capped
        self.assertLessEqual(calibrate.expected_half_width(0.05, 900), calibrate.TARGET_HALF_WIDTH)

    def test_planning_readout_reproduces_the_protocol_power_table(self):
        row = calibrate.planning(0.05)
        self.assertEqual(row["pairs_for_power_asymptotic"]["0.030"], 512)
        self.assertEqual(round(row["exact_power"]["512"]["0.030"], 2), 0.82)  # PROTOCOL.md section 8 table
        self.assertAlmostEqual(row["exact_power"]["768"]["0.030"], 0.95, delta=0.006)
        self.assertEqual(round(calibrate.planning(0.10)["exact_power"]["512"]["0.030"], 2), 0.55)

    def test_invalid_inputs(self):
        for value in (-0.1, 1.5):
            with self.assertRaises(calibrate.CalibrationError):
                calibrate.required_items(value)


class RequestTests(unittest.TestCase):
    def test_body_is_the_rig_body_with_stream_false(self):
        captured = {}

        class FakeConnection:
            def __init__(self, host, port, timeout):
                pass

            def connect(self):
                pass

            def request(self, method, path, body, headers):
                captured["body"] = body

            def getresponse(self):
                raise OSError("captured")

            def close(self):
                pass

        tokens = [151644, 1, 2, 3]
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(http_backend.http.client, "HTTPConnection", FakeConnection):
            payload = http_backend.LlamaCppCompletionPayload(
                request_id="r", expected_model_alias="alias", input_tokens=len(tokens),
                output_tokens=calibrate.OUTPUT_TOKENS, prompt_tokens=tuple(tokens), seed=17,
                stream_path=Path(root) / "s.raw", on_first_token=lambda _ns: None)
            with self.assertRaises(OSError):
                http_backend.LlamaCppHttpClient().complete("http://127.0.0.1:1", payload, lambda: None)
        rig = json.loads(captured["body"])
        ours = calibrate.request_body(tokens, 17)
        self.assertEqual(rig.pop("stream"), True)
        self.assertEqual(ours.pop("stream"), False)
        self.assertEqual(ours, rig)

    def test_response_must_carry_every_generated_token(self):
        tokens, content = answer_output(5)
        self.assertEqual(calibrate.response_output(response(tokens, content)), (tokens, content))
        for raw in (response(tokens[:-1], content), response(tokens, content, predicted=511),
                    b'{"content": 3, "tokens": []}'):
            with self.assertRaises(calibrate.CalibrationError):
                calibrate.response_output(raw)

    def test_float_tolerant_sensitivity_is_separate_from_the_frozen_rule(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "streams").mkdir()
            item = Gsm8kItem(split="train", index=1, question="q", reference=Decimal(46))
            tokens = list(range(calibrate.OUTPUT_TOKENS))
            (directory / "streams" / f"{item.item_id}.json").write_bytes(
                response(tokens, "Final answer: 46.00000000000001\n"))
            output = calibrate.load_outputs(calibrate.RunSpec("qwen", "q4", directory), [item])[item.item_id]
        self.assertFalse(output.correct)
        self.assertTrue(output.tolerant)
        self.assertFalse(calibrate.tolerant_match(Decimal("46.001"), Decimal(46)))
        self.assertTrue(calibrate.tolerant_match(Decimal("1340.0000000000001"), Decimal(1340)))
        self.assertFalse(calibrate.tolerant_match(None, Decimal(1)))
        summary = calibrate.format_summary([output])
        self.assertEqual((summary["correct"], summary["tolerant_correct"], summary["float_near_misses"]), (0, 1, 1))

    def test_prompt_structure_check(self):
        good = [*EXPECTED_FIRST_TOKENS["qwen"], 9, 9, *EXPECTED_LAST_TOKENS["qwen"]]
        calibrate.check_prompt_structure("qwen", good)
        with self.assertRaises(calibrate.CalibrationError):
            calibrate.check_prompt_structure("qwen", good[1:])


class PieceTableTests(unittest.TestCase):
    def test_byte_level_bpe_with_a_character_split_across_tokens(self):
        # GPT-2 byte-level: '\u0120' = space; ' \u2705' (E2 9C 85) split as '\u0120\u00e2\u013e' + '\u0127'
        texts = ["Final", "\u0120answer", "\u0120\u00e2\u013e", "\u0127", "<think>", "<|im_end|>", "[PAD0]"]
        table = calibrate.PieceTable(texts, [1, 1, 1, 1, 4, 3, 5], "gpt2")
        tokens = [0, 1, 2, 3, 4, 5, 6, 1]
        content = "Final answer \u2705<think> answer"
        starts = table.char_starts(tokens, content)
        self.assertEqual(starts, [(0, 0), (1, 5), (2, 12), (3, 13), (4, 14), (5, 21), (6, 21), (7, 21)])
        self.assertIsNone(table.char_starts(tokens, content + "!"))

    def test_spm_style_vocabulary_with_byte_tokens(self):
        table = calibrate.PieceTable(["<bos>", "\u2581Final", "\u2581answer", "<0xE2>", "<0x9C>", "<0x85>", ":"],
                                     [3, 1, 1, 6, 6, 6, 1], "gemma4")
        tokens = [0, 1, 2, 3, 4, 5, 6]
        content = " Final answer\u2705:"
        self.assertEqual(table.char_starts(tokens, content),
                         [(0, 0), (1, 0), (2, 6), (3, 13), (4, 13), (5, 13), (6, 14)])

    def test_offsets_drive_the_protocols_position_readouts(self):
        table = calibrate.PieceTable(["a", "Final answer: 18", "\n", "b"], [1, 1, 1, 1], "gemma4")
        tokens = [0, 1, 2] + [3] * (calibrate.OUTPUT_TOKENS - 3)
        content = "aFinal answer: 18\n" + "b" * (calibrate.OUTPUT_TOKENS - 3)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "streams").mkdir()
            item = Gsm8kItem(split="train", index=1, question="q", reference=Decimal(18))
            (directory / "streams" / f"{item.item_id}.json").write_bytes(response(tokens, content))
            output = calibrate.load_outputs(calibrate.RunSpec("gemma", "q4", directory), [item], table)[item.item_id]
            self.assertTrue(output.positions and output.correct)
            self.assertEqual(output.answer_token_end(), 2)
            unlocated = calibrate.load_outputs(calibrate.RunSpec("gemma", "q4", directory), [item])[item.item_id]
            self.assertFalse(unlocated.positions)
            summary = calibrate.format_summary([output, unlocated])
            self.assertEqual(summary["positions_available"], 1)
            self.assertEqual(summary["misses_by_budget_of_positioned"], {"256": 0, "320": 0, "384": 0, "512": 0})
            self.assertEqual(summary["no_answer_within_budget"], 0)


def write_run(directory, role, fmt, outputs, *, prompts_sha="sha256:p", flags=None):
    """outputs: {item_id: (tokens, content)}"""
    (directory / "streams").mkdir(parents=True)
    for item_id, (tokens, content) in outputs.items():
        (directory / "streams" / f"{item_id}.json").write_bytes(response(tokens, content))
    identity = {"role": role, "format": fmt, "server_flags": list(flags or calibrate.SERVER_FLAGS),
                "server_binaries_sha256": {"llama-server": "b"}, "output_tokens": calibrate.OUTPUT_TOKENS,
                "parallel_requests": calibrate.PARALLEL}
    (directory / "RUN.json").write_text(json.dumps({"identity": identity, "prompts_sha256": prompts_sha,
                                                    "complete_prefix_items": len(outputs), "sessions": []}))


class ReportTests(unittest.TestCase):
    """Two models x four formats with fabricated correctness patterns."""

    N = 20

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.items = train_items()
        self.sequences = calibrate.calibration_sequences(self.items)
        self.patch = mock.patch.object(calibrate, "load_items", lambda path, split: self.items)
        self.patch.start()
        # correct item positions per (role, format)
        self.correct = {
            ("qwen", "original"): set(range(0, 16)),
            ("qwen", "q4"): set(range(0, 14)) | {17},          # losses 14, 15; gain 17
            ("qwen", "dequant"): set(range(0, 14)) | {17, 18},   # gain 18 vs q4
            ("qwen", "dequant-repeat"): set(range(0, 14)) | {17},  # loss 18 vs dequant
            ("gemma", "original"): set(range(0, 10)),
            ("gemma", "q4"): set(range(0, 9)),                  # loss 9
            ("gemma", "dequant"): set(range(0, 9)),
            ("gemma", "dequant-repeat"): set(range(0, 9)) | {12},  # gain 12
        }
        self.specs = []
        for (role, fmt), correct in self.correct.items():
            outputs = {}
            for position, item in enumerate(self.sequences[role][:self.N]):
                value = item.reference if position in correct else item.reference + 1
                outputs[item.item_id] = answer_output(value)
            directory = self.root / f"{role}-{fmt}"
            write_run(directory, role, fmt, outputs)
            self.specs.append(calibrate.RunSpec(role, fmt, directory))

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def args(self, **overrides):
        values = dict(gsm8k=Path("unused"), run=self.specs, out=self.root / "r.json", tokenizer_model=[],
                      qwen_items=self.N, gemma_items=self.N)
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_paired_tables_intervals_and_pooling(self):
        report = calibrate.score(self.args())
        qwen = report["comparisons"]["qwen"]
        self.assertEqual((qwen["quantization"]["gain"], qwen["quantization"]["loss"]), (1, 2))
        self.assertEqual((qwen["execution-format"]["gain"], qwen["execution-format"]["loss"]), (1, 0))
        self.assertEqual((qwen["noise-floor"]["gain"], qwen["noise-floor"]["loss"]), (0, 1))
        self.assertEqual((qwen["rig-weights"]["gain"], qwen["rig-weights"]["loss"]), (2, 2))
        self.assertAlmostEqual(qwen["quantization"]["discordance"], 3 / self.N)
        self.assertAlmostEqual(qwen["quantization"]["difference"], -1 / self.N)
        pooled = report["pooled"]["quantization"]
        counts = PairedCounts(both=14 + 9, loss=2 + 1, gain=1, neither=3 + 10)
        self.assertEqual((pooled["both_correct"], pooled["loss"], pooled["gain"], pooled["neither_correct"]),
                         (counts.both, counts.loss, counts.gain, counts.neither))
        self.assertEqual(pooled["tango_95"], list(tango_interval(counts, 0.95)))
        self.assertEqual(pooled["newcombe_95"], list(newcombe_interval(counts, 0.95)))
        self.assertEqual(report["formats"]["qwen"]["original"]["correct"], 16)
        self.assertEqual(report["pooled_formats"]["q4"]["correct"], 15 + 9)
        self.assertEqual(report["formats"]["gemma"]["q4"]["extraction_rules"], {"final-answer": self.N})
        self.assertEqual(qwen["noise-floor"]["similarity"]["pairs"], self.N)
        self.assertIn("quantization", calibrate.markdown(report))
        self.assertEqual(set(report["planning"]), set(report["pooled"]))
        self.assertAlmostEqual(report["planning"]["quantization"]["discordance"], 4 / (2 * self.N))
        # exact integer answers: the tolerant sensitivity equals the frozen score here
        for key in ("gain", "loss", "both_correct"):
            self.assertEqual(pooled["tolerant_sensitivity"][key], pooled[key])

    def test_blinded_sizing_reports_discordance_only(self):
        with mock.patch.object(calibrate, "PILOT_ITEMS", self.N):
            value = calibrate.size(self.args())
        text = json.dumps(value)
        for word in ("accuracy", "gain", "loss", "difference", "correct"):
            self.assertNotIn(word, text)
        qwen = value["models"]["qwen"]
        self.assertEqual({name: row["discordant"] for name, row in qwen["discordance"].items()},
                         {"quantization": 3, "execution-format": 1, "noise-floor": 1})
        self.assertEqual(qwen["items"], calibrate.required_items(3 / self.N))

    def test_runs_that_differ_in_server_settings_are_refused(self):
        spec = self.specs[1]
        (spec.directory / "RUN.json").unlink()
        manifest = {"identity": {"role": spec.role, "format": spec.format, "server_flags": ["-np", "4"],
                                 "server_binaries_sha256": {"llama-server": "b"},
                                 "output_tokens": calibrate.OUTPUT_TOKENS, "parallel_requests": calibrate.PARALLEL},
                    "prompts_sha256": "sha256:p", "complete_prefix_items": self.N, "sessions": []}
        (spec.directory / "RUN.json").write_text(json.dumps(manifest))
        with self.assertRaises(calibrate.CalibrationError):
            calibrate.score(self.args())

    def test_incomplete_runs_are_refused(self):
        with self.assertRaises(calibrate.CalibrationError):
            calibrate.score(self.args(qwen_items=self.N + 1))


class FakeServer:
    """Stands in for LlamaServer: template-shaped tokens, full-budget answers, request log."""

    requests = []
    tokenizer_offset = 0

    def __init__(self, binary, model, gpu, port, log_path):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def get(self, path):
        return {"build_info": "b-test", "total_slots": calibrate.PARALLEL,
                "default_generation_settings": {"n_ctx": calibrate.SLOT_CONTEXT}}

    def tokenize(self, text):
        body = [len(text) + self.tokenizer_offset, sum(map(ord, text)) % 1000]
        return [*EXPECTED_FIRST_TOKENS["qwen"], *body, *EXPECTED_LAST_TOKENS["qwen"]]

    def post(self, path, body):
        FakeServer.requests.append(body)
        tokens, content = answer_output(body["seed"])
        return response(tokens, content)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.items = train_items()
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"weights")
        self.server = self.root / "bin" / "llama-server"
        self.server.parent.mkdir()
        self.server.write_bytes(b"binary")
        (self.server.parent / "libllama.so").write_bytes(b"library")
        FakeServer.requests = []
        FakeServer.tokenizer_offset = 0
        inventory = {"gpus": [{"index": "0", "uuid": "GPU-a", "name": "A6000", "memory_used_mib": 0, "driver": "x"},
                              {"index": "1", "uuid": "GPU-b", "name": "A6000", "memory_used_mib": 900, "driver": "x"}],
                     "compute_apps": [{"pid": 9, "gpu_uuid": "GPU-b", "used_memory_mib": "900"}]}
        self.patches = [mock.patch.object(calibrate, "load_items", lambda path, split: self.items),
                        mock.patch.object(calibrate, "LlamaServer", FakeServer),
                        mock.patch.object(calibrate, "gpu_inventory", lambda: inventory)]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.tmp.cleanup()

    def args(self, **overrides):
        values = dict(gsm8k=Path("unused"), role="qwen", format="q4", model=self.model, model_sha256=None,
                      verify_model_sha256=False, server=self.server, gpu="0", port=1, items=5,
                      prompts=self.root / "prompts.jsonl", write_prompts=True, allow_tokenizer_mismatch=False,
                      allow_busy_gpu=False, run_dir=self.root / "run-q4")
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_run_writes_outputs_prompts_and_resumes_only_missing_items(self):
        result = calibrate.run(self.args())
        self.assertEqual(result["complete_prefix_items"], 5)
        self.assertEqual(len(FakeServer.requests), 5)
        sequence = calibrate.calibration_sequences(self.items)["qwen"]
        prompts = calibrate.load_prompts(self.root / "prompts.jsonl")
        self.assertEqual(list(prompts), [item.item_id for item in sequence[:5]])
        self.assertEqual(FakeServer.requests[0], calibrate.request_body(prompts[sequence[0].item_id],
                                                                        sequence[0].index))
        calibrate.run(self.args(items=8))
        self.assertEqual(len(FakeServer.requests), 8)
        manifest = json.loads((self.root / "run-q4" / "RUN.json").read_text())
        self.assertEqual(manifest["complete_prefix_items"], 8)
        self.assertEqual(len(manifest["sessions"]), 2)
        self.assertEqual(manifest["identity"]["model_sha256"], calibrate.sha256_file(self.model))
        self.assertEqual(set(manifest["identity"]["server_binaries_sha256"]), {"llama-server", "libllama.so"})

    def test_other_formats_reuse_the_reference_prompts_and_refuse_tokenizer_drift(self):
        calibrate.run(self.args())
        calibrate.run(self.args(format="dequant", write_prompts=False, run_dir=self.root / "run-dq"))
        FakeServer.tokenizer_offset = 1
        with self.assertRaises(calibrate.CalibrationError):
            calibrate.run(self.args(format="original", write_prompts=False, run_dir=self.root / "run-orig"))
        FakeServer.tokenizer_offset = 0
        with self.assertRaises(calibrate.CalibrationError):  # reference prompts do not cover item 6..8
            calibrate.run(self.args(format="original", items=8, write_prompts=False,
                                    run_dir=self.root / "run-orig2"))

    def test_changed_settings_busy_gpu_and_missing_prompts_are_refused(self):
        calibrate.run(self.args())
        with self.assertRaises(calibrate.CalibrationError):  # same directory, different model hash
            calibrate.run(self.args(model_sha256="0" * 64))
        with self.assertRaises(calibrate.CalibrationError):
            calibrate.run(self.args(gpu="1", run_dir=self.root / "run-busy"))
        with self.assertRaises(calibrate.CalibrationError):
            calibrate.run(self.args(prompts=self.root / "absent.jsonl", write_prompts=False,
                                    run_dir=self.root / "run-np"))


class ServerLifecycleTests(unittest.TestCase):
    def test_failed_start_terminates_the_server_and_loading_503_is_waited_out(self):
        process = mock.Mock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(calibrate.subprocess, "Popen", return_value=process), \
                mock.patch.object(calibrate.LlamaServer, "_wait_healthy",
                                  side_effect=calibrate.CalibrationError("not healthy")):
            server = calibrate.LlamaServer(Path("llama-server"), Path("m.gguf"), "0", 1, Path(root) / "log")
            with self.assertRaises(calibrate.CalibrationError):
                with server:
                    pass
        process.terminate.assert_called_once()

        answers = iter([calibrate.CalibrationError("GET /health status 503"), {"status": "ok"}])

        def get(path):
            value = next(answers)
            if isinstance(value, Exception):
                raise value
            return value

        server = calibrate.LlamaServer(Path("llama-server"), Path("m.gguf"), "0", 1, Path("log"))
        server.process = process
        with mock.patch.object(server, "get", side_effect=get), mock.patch.object(calibrate.time, "sleep"):
            server._wait_healthy(timeout_s=60)


if __name__ == "__main__":
    unittest.main()
