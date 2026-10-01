"""GSM8K quality-suite builder: item assignment, pacing, campaign-loader compatibility and the separate key."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from research_dev.scheduler.campaigns.burstgpt import trace_inputs
from research_dev.scheduler.campaigns.burstgpt.build_realistic_trace import canonical
from research_dev.scheduler.campaigns.burstgpt.quality import build_trace, protocol
from research_dev.scheduler.campaigns.burstgpt.quality.gsm8k import QualityDataError

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import quality_fixtures as fx  # noqa: E402

ROLE_MODELS = {trace_inputs.QWEN_ROLE: fx.MODEL_IDS["qwen"], trace_inputs.GEMMA_ROLE: fx.MODEL_IDS["gemma"]}


class QualityBuildTraceTests(unittest.TestCase):
    def test_assignment_is_seeded_disjoint_and_ordered_by_role(self):
        items = fx.synthetic_items(30)
        per_shard = {"llama": 1, "gemma": 4, "qwen": 5}
        first = build_trace.assign_items(items, seed=11, shards=3, per_shard=per_shard)
        again = build_trace.assign_items(items, seed=11, shards=3, per_shard=per_shard)
        self.assertEqual(first, again)
        chosen = [item.index for shard in first for role in build_trace.ROLES for item in shard[role]]
        self.assertEqual(len(chosen), 30)
        self.assertEqual(len(set(chosen)), 30)
        self.assertEqual([len(first[0][role]) for role in build_trace.ROLES], [1, 4, 5])
        self.assertNotEqual(first, build_trace.assign_items(items, seed=12, shards=3, per_shard=per_shard))
        with self.assertRaises(QualityDataError):
            build_trace.assign_items(items, seed=11, shards=4, per_shard=per_shard)

    def test_protocol_shards_fit_the_test_split(self):
        size = protocol.LLAMA_PER_SHARD + protocol.GEMMA_PER_SHARD + protocol.QWEN_PER_SHARD
        self.assertLessEqual(protocol.MAXIMUM_SHARDS * size, protocol.GSM8K_ROWS["test"])
        self.assertLessEqual(protocol.INTERIM_SHARDS, protocol.PLANNED_SHARDS)
        self.assertLessEqual(protocol.PLANNED_SHARDS, protocol.MAXIMUM_SHARDS)

    def test_arrivals_pace_each_block_below_its_batched_service(self):
        plan = build_trace.arrival_plan({"llama": 2, "gemma": 32, "qwen": 32}, {"llama": 384, "gemma": 384, "qwen": 384},
                                        build_trace.DEFAULT_SERVICE_S_PER_TOKEN, build_trace.DEFAULT_BATCH_ROWS)
        self.assertEqual(plan["llama"], [1_000_000, 2_000_000])
        self.assertEqual(plan["gemma"][0], 5_000_000)
        gemma_gap = plan["gemma"][1] - plan["gemma"][0]
        qwen_gap = plan["qwen"][1] - plan["qwen"][0]
        self.assertEqual(gemma_gap, round(0.8 * 384 * 0.40 / 2 * 1e6))
        self.assertEqual(qwen_gap, round(0.8 * 384 * 0.50 / 4 * 1e6))
        gemma_service = 32 * 384 * 0.40 / 2 * 1e6
        self.assertEqual(plan["qwen"][0], 5_000_000 + int(0.9 * gemma_service))
        self.assertGreater(plan["qwen"][0], plan["gemma"][-1])
        merged = plan["llama"] + plan["gemma"] + plan["qwen"]
        self.assertEqual(merged, sorted(merged))

    def test_suite_shards_load_through_the_campaign_trace_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "suite"
            suite = fx.build_small_suite(out, shards=2)
            self.assertEqual(len(suite["shards"]), 2)
            self.assertEqual([entry["planned"] for entry in suite["shards"]], [True, False])
            for entry in suite["shards"]:
                directory = Path(entry["directory"])
                manifest = json.loads((directory / "TRACE_MANIFEST.json").read_text(encoding="ascii"))
                large, overlay = trace_inputs.validate_trace(
                    directory / "REQUESTS_SEMANTIC_SOURCE.jsonl", directory / "REQUESTS_OVERLAY.jsonl", manifest)
                self.assertEqual((len(large), len(overlay)), (6, 1))
                merged = trace_inputs.merge_rows(large, overlay, ROLE_MODELS)
                schedule = json.loads((directory / f"{entry['trace_name']}.json").read_text(encoding="ascii"))
                selected, replay = trace_inputs.apply_named_replay_schedule(merged, schedule)
                models = [item["model_id"] for item in selected]
                self.assertEqual(models, [fx.MODEL_IDS["llama"]] + [fx.MODEL_IDS["gemma"]] * 3 + [fx.MODEL_IDS["qwen"]] * 3)
                self.assertEqual(replay["trace_name"], entry["trace_name"])
                self.assertEqual(manifest["model_inventory"], fx.INVENTORY)
                self.assertEqual(manifest["derivation"]["quality_protocol"], protocol.PROTOCOL_ID)
                self.assertTrue(all(row["output_tokens"] == 12 and row["slo_us"] == 30_000_000 for row in large + overlay))
                for name, sha in entry["files"].items():
                    self.assertEqual(sha, build_trace.digest_file(directory / name))

    def test_answers_stay_out_of_the_traces_and_the_key_binds_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "suite"
            fx.build_small_suite(out, shards=2)
            key = fx.key_rows(out)
            self.assertEqual(len(key), 14)
            rows = {}
            for shard in ("shard-00", "shard-01"):
                for name in ("REQUESTS_SEMANTIC_SOURCE.jsonl", "REQUESTS_OVERLAY.jsonl"):
                    text = (out / shard / name).read_text(encoding="ascii")
                    self.assertNotIn("reference", text)
                    self.assertNotIn("gsm8k-test:", text)
                    for line in text.splitlines():
                        row = json.loads(line)
                        rows[row["event_id"]] = row
            for entry in key:
                row = rows[entry["request_id"]]
                self.assertEqual(entry["prompt_sha256"], "sha256:" + hashlib.sha256(
                    canonical(row["prompt_tokens"])).hexdigest())
                self.assertEqual(entry["input_tokens"], len(row["prompt_tokens"]))
                index = int(entry["item_id"].split(":")[1])
                self.assertEqual(entry["reference"], str(index + 1))
            prompts = [json.loads(line) for line in (out / "QUALITY_PROMPTS.jsonl").read_text().splitlines()]
            self.assertEqual({row["request_id"] for row in prompts}, {row["request_id"] for row in key})
            self.assertTrue(all("Final answer: <number>" in row["prompt_text"] for row in prompts))
            suite = json.loads((out / "SUITE.json").read_text())
            self.assertEqual(suite["key_sha256"], build_trace.digest_file(out / "QUALITY_KEY.jsonl"))

    def test_codec_output_without_the_template_structure_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            codecs = {role: fx.FakeCodec(role, broken=(role == "gemma")) for role in build_trace.ROLES}
            with self.assertRaisesRegex(QualityDataError, "gemma prompt tokens"):
                build_trace.build_suite(
                    items=fx.synthetic_items(10), dataset={"split": "test"}, output_dir=Path(tmp) / "s",
                    trace_prefix="q", codecs=codecs, inventory=fx.INVENTORY, seed=1, shards=1,
                    per_shard={"llama": 1, "gemma": 1, "qwen": 1},
                    output_tokens={role: 8 for role in build_trace.ROLES}, planned_shards=1, tokenizers={})

    def test_existing_suite_and_missing_roles_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "suite"
            fx.build_small_suite(out, shards=1)
            with self.assertRaises(QualityDataError):
                fx.build_small_suite(out, shards=1)
            with self.assertRaisesRegex(QualityDataError, "three-model"):
                fx.build_small_suite(Path(tmp) / "other", shards=1, per_shard={"llama": 0, "gemma": 2, "qwen": 2})

    def test_inventory_is_copied_only_when_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "TRACE_MANIFEST.json"
            path.write_text(json.dumps({"model_inventory": fx.INVENTORY}))
            self.assertEqual(build_trace.load_inventory(path), fx.INVENTORY)
            partial = {name: row for name, row in fx.INVENTORY.items() if "llama" not in name}
            path.write_text(json.dumps({"model_inventory": partial}))
            with self.assertRaises(QualityDataError):
                build_trace.load_inventory(path)


if __name__ == "__main__":
    unittest.main()
