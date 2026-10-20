"""WS24 traces: WildChat conversation builder, Mooncake long-prompt builder, fetch tool, cache_idle_slots flag."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_dev.scheduler import GGUFModelManifestLoader
from research_dev.scheduler.adapters import (
    LlamaServerProcessConfiguration, LlamaServerProcessLauncher, PhysicalAdapterError, llama_server_launch_contract,
)
from research_dev.scheduler.adapters.llama_server import _launch_contract_supports_execution
from research_dev.scheduler.campaigns.burstgpt import build_conversation_trace as conv
from research_dev.scheduler.campaigns.burstgpt import build_longprompt_trace as lp
from research_dev.scheduler.campaigns.burstgpt import ws24_trace_common as common
from research_dev.scheduler.campaigns.burstgpt.build_realistic_trace import canonical
from research_dev.scheduler.campaigns.burstgpt.tools import fetch_wildchat_sample as fetch
from research_dev.scheduler.campaigns.burstgpt.trace_inputs import (
    GEMMA_ROLE, QWEN_ROLE, REPLAY_START_US, apply_named_replay_schedule, merge_rows, validate_trace,
)
from test_gguf_cost import write_synthetic_gguf
from test_llama_server_adapter import execution_command

PIECE = re.compile(r"<\|[a-z_]+\|>|<\|[a-z]+>|<[a-z]+\|>|\S+|\n")
LLAMA_SHA = "4b" * 32
INVENTORY = {"gemma-4-12b-q40-dequant-f16": {"artifact_bytes": 10, "artifact_file": "g.gguf", "artifact_sha256": "a" * 64, "kind": "text_decoder"},
             "qwen3-14b-q4km-dequant-f16": {"artifact_bytes": 10, "artifact_file": "q.gguf", "artifact_sha256": "b" * 64, "kind": "text_decoder"},
             "llama-3.2-1b-instruct-q4_0": {"artifact_bytes": 10, "artifact_file": "l.gguf", "artifact_sha256": LLAMA_SHA, "kind": "text_decoder"}}


class FakeCodec:
    """Deterministic word-level tokenizer; prepends ``bos`` like the real codec (add_special) when configured."""

    def __init__(self, bos=None):
        self.bos = bos
        self.calls = 0

    def tokenize(self, text):
        self.calls += 1
        tokens = [int(hashlib.md5(piece.encode()).hexdigest()[:6], 16) % 50000 + 1000 for piece in PIECE.findall(text)]
        if not tokens:
            raise RuntimeError("empty")
        return ([self.bos] if self.bos is not None else []) + tokens

    def validate_detokenize(self, tokens):
        assert tokens

    def close(self):
        pass


def codecs():
    return {common.HOT_ROLE: FakeCodec(None), common.COLD_ROLE: FakeCodec(2), "overlay": FakeCodec(128000)}


def stamp(minute, second=0):
    return f"2023-04-09T00:{minute:02d}:{second:02d}Z"


def conversation(name, turns, words=12, reply_words=20, model="gpt-4-0314", gap_min=2, language="English", toxic=False):
    messages = []
    for turn in range(turns):
        messages.append({"role": "user", "content": " ".join(f"{name}u{turn}w{i}" for i in range(words)), "timestamp": None})
        messages.append({"role": "assistant", "content": " ".join(f"{name}a{turn}w{i}" for i in range(reply_words)),
                         "timestamp": stamp(5 + turn * gap_min)})
    return {"conversation_hash": name, "model": model, "timestamp": stamp(5), "turn": turns, "language": language,
            "toxic": toxic, "redacted": False, "conversation": messages}


def sample():
    return [conversation("c1", 3), conversation("c2", 4, words=8, reply_words=30, model="gpt-3.5-turbo-0301", gap_min=3),
            conversation("c3", 3, words=20), conversation("c4", 2), conversation("toxic", 3, toxic=True),
            conversation("de", 3, language="German")]


class ConversationTraceTests(unittest.TestCase):
    def build(self, **changes):
        options = conv.ConversationOptions(**{"trace_name": "t", **changes})
        return conv.build_conversation_trace(sample(), codecs(), options, llama_bytes=10, llama_sha=LLAMA_SHA)

    def test_turn_prompts_are_token_exact_prefixes_paced_sequentially_and_deterministic(self):
        large, overlay, stats = self.build()
        again, _, _ = self.build()
        self.assertEqual(b"".join(canonical(r) for r in large), b"".join(canonical(r) for r in again))
        self.assertEqual(stats["conversations"], 3)                      # c4 (2 turns), toxic and German are excluded
        self.assertEqual(set(stats["selected_conversations"]), {"c1", "c2", "c3"})
        by_conversation = {}
        for row in large:
            by_conversation.setdefault(row["conversation_id"], []).append(row)
        for rows in by_conversation.values():
            for previous, current in zip(rows, rows[1:]):
                self.assertEqual(current["prompt_tokens"][:len(previous["prompt_tokens"])], previous["prompt_tokens"])
                self.assertEqual(current["history_tokens"], len(previous["prompt_tokens"]))
                self.assertEqual(current["history_tokens"] + current["new_tokens"], current["input_tokens"])
                self.assertGreaterEqual(current["think_s"], 15.0)
                self.assertGreaterEqual(current["arrival_us"] - previous["arrival_us"],
                                        int((previous["service_estimate_s"] + current["think_s"]) * 1e6) - 1000)
            self.assertEqual(rows[0]["turn_index"], 0)
            self.assertEqual(rows[0]["history_tokens"], 0)
            self.assertEqual({r["turns_total"] for r in rows}, {len(rows)})
        for previous, current in zip(large, large[1:]):
            self.assertGreaterEqual(current["arrival_us"], previous["arrival_us"] + int(previous["service_estimate_s"] * 1e6) - 1000)
        self.assertEqual(large[0]["arrival_us"], REPLAY_START_US)
        self.assertEqual([r["request_index"] for r in large], list(range(len(large))))
        self.assertTrue(all(r["input_tokens"] == len(r["prompt_tokens"]) for r in large))
        self.assertGreater(stats["roles"][common.HOT_ROLE], 0)
        self.assertGreater(stats["roles"][common.COLD_ROLE], 0)
        self.assertEqual(large[0]["model_id"], common.TEMPLATES[common.HOT_ROLE].role_model_id)   # first conversation hot
        cold = [r for r in large if r["model_id"] == common.TEMPLATES[common.COLD_ROLE].role_model_id]
        self.assertEqual(cold[0]["prompt_tokens"][0], 2)                     # Gemma BOS once, at the front only
        self.assertEqual(cold[1]["prompt_tokens"].count(2), cold[0]["prompt_tokens"].count(2))
        self.assertEqual(len(overlay), 1)
        self.assertEqual(overlay[0]["conversation_id"], "c4")                 # the shortest unused first message
        self.assertEqual(overlay[0]["prompt_tokens"][0], 128000)
        self.assertEqual((overlay[0]["model_artifact_sha256"], overlay[0]["model_artifact_bytes"]), (LLAMA_SHA, 10))
        self.assertGreater(overlay[0]["arrival_us"], large[0]["arrival_us"])
        self.assertGreater(stats["prefix_reusable_fraction"], 0.4)

    def test_think_time_comes_from_assistant_timestamps(self):
        self.assertEqual(common.think_time_s(stamp(5), stamp(7), 1200), (100.0, "timestamps"))   # 120 s - 1200/60
        self.assertEqual(common.think_time_s(stamp(5), stamp(5, 20), 1200), (15.0, "timestamps"))  # clipped low
        self.assertEqual(common.think_time_s(stamp(5), stamp(25), 0), (300.0, "timestamps"))      # clipped high
        self.assertEqual(common.think_time_s(None, stamp(7), 100), (90.0, "default"))
        self.assertEqual(common.think_time_s(stamp(7), stamp(5), 100), (90.0, "default"))
        large, _, stats = self.build()
        self.assertEqual(stats["think_sources"], {"session start": 3, "timestamps": len(large) - 3})
        c2 = [r for r in large if r["conversation_id"] == "c2"]
        self.assertAlmostEqual(c2[1]["think_s"], 180.0 - len(c2[1]["prompt_tokens"]) * 0 - 30 * len("c2a1w00 ") / 60, delta=3)

    def test_context_cap_truncates_a_conversation(self):
        large, _, _ = self.build()
        c3 = [r for r in large if r["conversation_id"] == "c3"]
        self.assertEqual(len(c3), 3)
        cap = c3[2]["input_tokens"] + c3[2]["output_tokens"] - 1
        large, _, stats = self.build(context_cap=cap)
        c3 = [r for r in large if r["conversation_id"] == "c3"]
        self.assertEqual(len(c3), 2)
        self.assertEqual(c3[0]["conversation_truncated_by"], "context_cap")
        self.assertEqual({r["turns_total"] for r in c3}, {2})
        self.assertGreaterEqual(stats["truncated"]["context_cap"], 1)
        self.assertTrue(all(r["input_tokens"] + r["output_tokens"] <= cap for r in large))

    def test_duration_cuts_the_trace_and_two_conversations_are_required(self):
        large, _, stats = self.build()
        second_start = min(r["arrival_us"] for r in large if r["conversation_id"] == stats["selected_conversations"][1])
        short, _, short_stats = self.build(duration_s=(second_start - REPLAY_START_US) / 1e6 + 120.0)
        self.assertEqual(short_stats["conversations"], 2)
        self.assertLess(len(short), len(large))
        self.assertTrue(all(r["arrival_us"] + r["service_estimate_s"] * 1e6 <= REPLAY_START_US + short_stats["replay_span_s"] * 1e6
                            + 1e9 for r in short))
        with self.assertRaises(SystemExit):
            self.build(duration_s=30.0)

    def test_loader_accepts_the_written_trace(self):
        large, overlay, stats = self.build()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "trace"
            manifest = common.write_trace(out, "t", large, overlay, INVENTORY, {"builder": "test"})
            loaded_large, loaded_overlay = validate_trace(out / "REQUESTS_SEMANTIC_SOURCE.jsonl", out / "REQUESTS_OVERLAY.jsonl",
                                                          json.loads((out / "TRACE_MANIFEST.json").read_text()))
            self.assertEqual((len(loaded_large), len(loaded_overlay)), (len(large), 1))
            self.assertEqual(manifest["base_trace"]["roles"], {QWEN_ROLE: stats["roles"]["hot"], GEMMA_ROLE: stats["roles"]["cold"]})
            merged = merge_rows(loaded_large, loaded_overlay, {QWEN_ROLE: "qwen-exec", GEMMA_ROLE: "gemma-exec"})
            selected, schedule = apply_named_replay_schedule(merged, json.loads((out / "t.json").read_text()))
            self.assertEqual(len(selected), len(large) + 1)
            self.assertEqual(schedule["trace_name"], "t")
            self.assertEqual(schedule["replay_first_arrival_us"], REPLAY_START_US)
            with self.assertRaises(SystemExit):
                common.write_trace(out, "t", large, overlay, INVENTORY, {})


def mooncake_rows():
    rows = []
    for index in range(40):
        rows.append({"timestamp": index * 1000, "input_length": 1500 + index * 230, "output_length": 20 + index * 15,
                     "hash_ids": [0, index + 1]})
    rows.append({"timestamp": 50000, "input_length": 30000, "output_length": 100, "hash_ids": [0, 99]})
    return rows


class LongPromptTraceTests(unittest.TestCase):
    def build(self, **changes):
        options = lp.LongPromptOptions(**{"trace_name": "lp", **changes})
        return lp.build_longprompt_trace(mooncake_rows(), codecs(), options, llama_bytes=10, llama_sha=LLAMA_SHA)

    def test_fresh_variant_keeps_source_lengths_and_paces_sequentially(self):
        large, overlay, stats = self.build(variant="fresh", requests=8)
        self.assertEqual(len(large), 8)
        self.assertTrue(all(r["input_tokens"] == r["source_input_tokens"] == len(r["prompt_tokens"]) for r in large))
        self.assertTrue(all(2000 <= r["input_tokens"] <= 8000 and r["output_tokens"] <= 512 for r in large))
        self.assertTrue(all(r["history_tokens"] == 0 and r["new_tokens"] == r["input_tokens"] for r in large))
        for previous, current in zip(large, large[1:]):
            self.assertAlmostEqual(current["arrival_us"] - previous["arrival_us"], (previous["service_estimate_s"] + 60.0) * 1e6, delta=1000)
        self.assertEqual(stats["prefix_reusable_fraction"], 0.0)
        self.assertGreater(stats["roles"][common.HOT_ROLE], 0)
        self.assertGreater(stats["roles"][common.COLD_ROLE], 0)
        self.assertEqual(len(overlay), 1)
        self.assertEqual(overlay[0]["input_tokens"], 48)
        again, _, _ = self.build(variant="fresh", requests=8)
        self.assertEqual(b"".join(canonical(r) for r in large), b"".join(canonical(r) for r in again))

    def test_cached_variant_shares_a_token_exact_document_per_role(self):
        large, _, stats = self.build(variant="cached", requests=10, document_tokens=1500)
        for role in (common.HOT_ROLE, common.COLD_ROLE):
            rows = [r for r in large if r["model_id"] == common.TEMPLATES[role].role_model_id]
            self.assertGreater(len(rows), 0)
            heads = {tuple(r["prompt_tokens"][:1500]) for r in rows}
            self.assertEqual(len(heads), 1)
            self.assertEqual(rows[0]["history_tokens"], 0)
            self.assertTrue(all(r["history_tokens"] == 1500 for r in rows[1:]))
            tails = {tuple(r["prompt_tokens"][1500:]) for r in rows}
            self.assertEqual(len(tails), len(rows))                       # unique questions
        cold = [r for r in large if r["model_id"] == common.TEMPLATES[common.COLD_ROLE].role_model_id]
        self.assertEqual(cold[0]["prompt_tokens"][0], 2)
        self.assertEqual(cold[0]["prompt_tokens"].count(2), 1)
        self.assertTrue(all(r["input_tokens"] == r["source_input_tokens"] for r in large if not r["question_padded_to_minimum"]))
        self.assertGreater(stats["prefix_reusable_fraction"], 0.2)
        with self.assertRaises(SystemExit):
            self.build(variant="bogus")
        with self.assertRaises(SystemExit):
            self.build(variant="fresh", requests=100)


class FetchToolTests(unittest.TestCase):
    def test_offline_filter_keeps_clean_english_multiturn_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "rows_0.json"
            page.write_text(json.dumps({"rows": [{"row": r} for r in sample()] + [{"row": sample()[0]}]}))
            out = Path(directory) / "sample.json"
            fetch.main(["--from-fetched", str(page), "--output", str(out), "--min-turns", "3"])
            kept = json.loads(out.read_text())
            self.assertEqual([r["conversation_hash"] for r in kept], ["c1", "c2", "c3"])
            provenance = json.loads(out.with_suffix(".provenance.json").read_text())
            self.assertEqual((provenance["rows_seen"], provenance["rows_kept"], provenance["api"]), (7, 3, "offline"))
            self.assertEqual(provenance["sample_sha256"], hashlib.sha256(out.read_bytes()).hexdigest())
            fetch.main(["--from-fetched", str(page), "--output", str(out), "--model", "gpt-4-0314"])
            self.assertEqual([r["conversation_hash"] for r in json.loads(out.read_text())], ["c1", "c3"])


class CacheIdleSlotsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        model = self.root / "model.gguf"
        write_synthetic_gguf(model)
        self.manifest = GGUFModelManifestLoader.load("synthetic-model-id", model)
        self.command = execution_command(self.manifest.artifact_sha256)

    def test_contract_is_opt_in_and_identity_relevant(self):
        default = llama_server_launch_contract(self.command, self.manifest)
        self.assertFalse(default.cache_idle_slots)
        enabled = llama_server_launch_contract(replace(
            self.command, adapter_parameters={**self.command.adapter_parameters, "cache_idle_slots": True}), self.manifest)
        self.assertTrue(enabled.cache_idle_slots)
        self.assertFalse(_launch_contract_supports_execution(default, enabled))
        self.assertFalse(_launch_contract_supports_execution(enabled, default))
        # the integer spelling is what a model's runtime_parameters must use: the runtime catalog's adapter
        # parameters are int|str only (RuntimeModelEndpointCapability rejects JSON booleans)
        for value, expected in ((1, True), (0, False)):
            with self.subTest(value=value):
                self.assertIs(llama_server_launch_contract(replace(
                    self.command, adapter_parameters={**self.command.adapter_parameters, "cache_idle_slots": value}),
                    self.manifest).cache_idle_slots, expected)
        for bad in ("yes", 2, -1, "true", "1", None):
            with self.subTest(bad=bad), self.assertRaises(PhysicalAdapterError):
                llama_server_launch_contract(replace(
                    self.command, adapter_parameters={**self.command.adapter_parameters, "cache_idle_slots": bad}), self.manifest)
        with self.assertRaises(PhysicalAdapterError):
            replace(default, cache_idle_slots="true")

    def test_completion_requests_reuse_the_prompt_cache_only_for_idle_slot_endpoints(self):
        """2026-10-05 rig (p9conv1nph): the body sent cache_prompt=false for every request, so the server
        never consulted the idle-slot prompt cache (0 % prefix reuse). The flag now follows the endpoint's
        cache_idle_slots adapter parameter; the paper arms (no flag) keep the full prefill."""
        from research_dev.scheduler.adapters.http_backend import (
            LlamaCppCompletionPayload, LlamaCppHttpClient, prompt_cache_requested,
        )
        payload = LlamaCppCompletionPayload(
            request_id="r", expected_model_alias="m", input_tokens=2, output_tokens=3, prompt_tokens=(1, 2),
            seed=1, stream_path=self.root / "body.raw", on_first_token=lambda _: None)
        self.assertFalse(LlamaCppHttpClient._completion_request_body(payload)["cache_prompt"])
        self.assertTrue(LlamaCppHttpClient._completion_request_body(payload, True)["cache_prompt"])
        self.assertEqual(LlamaCppHttpClient._completion_request_body(payload, True)["prompt"], [1, 2])
        self.assertFalse(prompt_cache_requested(self.command))
        self.assertFalse(prompt_cache_requested(replace(
            self.command, adapter_parameters={**self.command.adapter_parameters, "cache_idle_slots": 0})))
        for on in (1, True):
            self.assertTrue(prompt_cache_requested(replace(
                self.command, adapter_parameters={**self.command.adapter_parameters, "cache_idle_slots": on})))
        with self.assertRaises(PhysicalAdapterError):
            LlamaCppHttpClient().complete("http://127.0.0.1:1", payload, lambda: None, cache_prompt=1)
        # the accounting check: cached prompt tokens count only when the request asked for the cache
        ok = LlamaCppHttpClient._prompt_accounting_ok
        self.assertTrue(ok({"prompt_n": 2500}, 2500, False))
        self.assertTrue(ok({"prompt_n": 2500, "cache_n": 0}, 2500, True))
        self.assertTrue(ok({"prompt_n": 497, "cache_n": 2003}, 2500, True))
        self.assertFalse(ok({"prompt_n": 497, "cache_n": 2003}, 2500, False))   # reuse without asking for it
        self.assertFalse(ok({"prompt_n": 496, "cache_n": 2003}, 2500, True))    # a token unaccounted for
        self.assertFalse(ok({"prompt_n": 0, "cache_n": 2500}, 2500, True))      # the server always prefills one
        self.assertFalse(ok({"cache_n": 2500}, 2500, True))

    def test_launch_omits_no_cache_idle_slots_only_when_enabled(self):
        path = self.root / "llama-server"
        path.write_text("#!/bin/sh\n", encoding="ascii")
        path.chmod(0o755)
        launcher = LlamaServerProcessLauncher(LlamaServerProcessConfiguration(
            server_path=path, model_paths_by_artifact={self.manifest.artifact_sha256: self.root / "model.gguf"},
            library_paths_by_device={}, executable_device_names={}, output_directory=self.root))
        contract = replace(llama_server_launch_contract(self.command, self.manifest),
                           gpu_layers=0, phone_device_id=None, ffn_environment={})

        class FakeServer:
            def __init__(self, command, environment, directory, label, launch_contract):
                self.command, self.environment, self.launch_contract = command, environment, launch_contract
                self.stderr_lines = []
                self.process = SimpleNamespace(poll=lambda: None)

            def start(self):
                pass

            def stop(self):
                pass

        with patch("research_dev.scheduler.adapters.llama_server.ManagedLlamaServer", FakeServer), \
                patch.object(launcher, "_healthy", return_value=True):
            default = launcher.launch_contract("http://127.0.0.1:19000", contract, self.manifest, label="d", control_check=lambda: None)
            enabled = launcher.launch_contract("http://127.0.0.1:19000", replace(contract, cache_idle_slots=True), self.manifest,
                                               label="e", control_check=lambda: None)
        self.assertIn("--no-cache-idle-slots", default.command)
        self.assertNotIn("--no-cache-idle-slots", enabled.command)
        self.assertIn("--kv-unified", enabled.command)
        self.assertEqual([a for a in default.command if a != "--no-cache-idle-slots"], list(enabled.command))


if __name__ == "__main__":
    unittest.main()
