"""Quality scorer on fabricated campaign runs with known outcomes (synthetic arms, no hardware)."""
import argparse
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from research_dev.scheduler.campaigns.burstgpt.quality import protocol, score
from research_dev.scheduler.campaigns.burstgpt.quality.stats import PairedCounts, noninferiority

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import quality_fixtures as fx  # noqa: E402

BUDGET = 12


def reference(row):
    return row["reference"]


class QualityScoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.suite_dir = self.root / "suite"
        fx.build_small_suite(self.suite_dir, shards=2, output_tokens=BUDGET)
        self.key = fx.key_rows(self.suite_dir)
        self.confirmatory = sorted((row for row in self.key if row["role"] in ("qwen", "gemma")),
                                   key=lambda row: row["request_id"])
        self.traces = sorted({row["trace_name"] for row in self.key})
        # A: baseline wrong, treatment right (gain); C: baseline right, treatment wrong (loss);
        # B: same answer, the treatment diverges only after it
        self.a, self.b, self.c = (row["request_id"] for row in self.confirmatory[:3])

    def tearDown(self):
        self.tmp.cleanup()

    def baseline_outputs(self):
        outputs = {}
        for row in self.key:
            value = str(int(reference(row)) + 1) if row["request_id"] == self.a else reference(row)
            outputs[row["request_id"]] = fx.answer_pieces(value, BUDGET)
        return outputs

    def paper_outputs(self):
        outputs = self.baseline_outputs()
        rows = {row["request_id"]: row for row in self.key}
        outputs[self.a] = fx.answer_pieces(reference(rows[self.a]), BUDGET)
        outputs[self.c] = fx.answer_pieces(str(int(reference(rows[self.c])) + 7), BUDGET)
        outputs[self.b] = fx.answer_pieces(reference(rows[self.b]), BUDGET, tail=" other")
        return outputs

    def write_arm(self, label, outputs, *, coverage=0.0, chunk=1, **kwargs):
        dirs = []
        for trace in self.traces:
            run = self.root / label / trace / "run"
            fx.write_run(run, trace, self.key, outputs,
                         coverage={row["request_id"]: coverage for row in self.key if row["role"] != "llama"},
                         chunk=chunk, **kwargs)
            dirs.append(f"{label}={run}")
        return dirs

    def namespace(self, arms, treatments, *, strict=(), aa=None, shards=2):
        return argparse.Namespace(suite=self.suite_dir / "SUITE.json", key=self.suite_dir / "QUALITY_KEY.jsonl",
                                  arm=arms, baseline="legacy", treatment=list(treatments), shards=shards,
                                  strict_arm=list(strict), aa=aa, bootstrap_replicates=200)

    def test_known_outcomes_give_the_expected_table_test_and_similarity(self):
        arms = (self.write_arm("legacy", self.baseline_outputs())
                + self.write_arm("paper", self.paper_outputs(), coverage=0.5, chunk=3)
                + self.write_arm("forced", self.baseline_outputs(), coverage=1.0)
                + self.write_arm("repeat", self.baseline_outputs()))
        report = score.score(self.namespace(arms, ["paper", "forced"], strict=["forced"], aa="repeat"))
        paper, forced = report["comparisons"]
        self.assertEqual((paper["primary"]["n"], paper["primary"]["gain"], paper["primary"]["loss"],
                          paper["primary"]["both_correct"]), (12, 1, 1, 10))
        expected = noninferiority(PairedCounts(both=10, loss=1, gain=1, neither=0), protocol.MARGIN,
                                  protocol.ALPHA_ONE_SIDED)
        self.assertEqual(paper["primary"]["interval"], expected["interval"])
        self.assertEqual(paper["primary"]["noninferior"], expected["noninferior"])
        self.assertEqual(paper["losses"], [next(r["item_id"] for r in self.key if r["request_id"] == self.c)])
        self.assertEqual(paper["gains"], [next(r["item_id"] for r in self.key if r["request_id"] == self.a)])
        similarity = paper["similarity"]
        self.assertEqual(similarity["pairs"], 12)
        self.assertAlmostEqual(similarity["identical_share"], 9 / 12)
        self.assertAlmostEqual(similarity["identical_through_answer_share"], 10 / 12)
        self.assertAlmostEqual(similarity["same_answer_share"], 10 / 12)
        # answer pieces: lead . \n Final answer : value -> the value is piece 6, the late divergence piece 8
        self.assertEqual(similarity["first_divergence_of_diverged"]["min"], 6)
        self.assertEqual(similarity["first_divergence_of_diverged"]["max"], 8)
        self.assertEqual(forced["primary"]["gain"] + forced["primary"]["loss"], 0)
        self.assertEqual(forced["similarity"]["identical_share"], 1.0)
        self.assertAlmostEqual(paper["coverage"]["token_weighted_assisted_share"], round(0.5 * 11) / 11)
        self.assertEqual(forced["coverage"]["token_weighted_assisted_share"], 1.0)
        self.assertTrue(forced["validity_checks"]["strict_arm_coverage"])
        self.assertNotIn("strict_arm_coverage", paper["validity_checks"])
        self.assertEqual(report["arms"]["legacy"]["accuracy"]["llama"]["accuracy_present"], 1.0)
        self.assertAlmostEqual(report["arms"]["legacy"]["pooled"]["accuracy_present"], 11 / 12)
        self.assertEqual(paper["control"]["similarity"]["identical_share"], 1.0)
        self.assertEqual(report["baseline_repeat"]["discordant"], 0)
        steps = report["conclusion"]["steps"]
        self.assertEqual(steps[0]["status"], "NONINFERIOR" if expected["noninferior"] else "NOT_SHOWN")
        if not expected["noninferior"]:
            self.assertEqual(steps[1]["status"], "NOT_TESTED")
        self.assertIn("| paper |", score.markdown(report))
        json.dumps(report, allow_nan=False)

    def test_missing_outputs_use_worst_case_imputation_and_invalidate_the_comparison(self):
        treatment = self.baseline_outputs()
        dropped = self.confirmatory[5]["request_id"]
        treatment[dropped] = None
        arms = self.write_arm("legacy", self.baseline_outputs()) + self.write_arm("paper", treatment)
        report = score.score(self.namespace(arms, ["paper"]))
        comparison = report["comparisons"][0]
        self.assertEqual(comparison["primary"]["loss"], 1)
        self.assertEqual(comparison["sensitivity"]["complete_case"]["n"], 11)
        self.assertEqual(comparison["sensitivity"]["complete_case"]["loss"], 0)
        self.assertFalse(comparison["validity_checks"]["missing_share_within_limit"])
        self.assertEqual(report["conclusion"]["steps"][0]["status"], "INVALID")
        self.assertEqual(report["arms"]["paper"]["pooled"]["missing"], 1)

    def test_incomplete_or_rejected_requests_are_missing(self):
        treatment = self.baseline_outputs()
        short = self.confirmatory[4]["request_id"]
        treatment[short] = treatment[short][:BUDGET - 1]
        rejected = self.confirmatory[7]["request_id"]
        arms = self.write_arm("legacy", self.baseline_outputs()) + self.write_arm("paper", treatment,
                                                                                  rejected=(rejected,))
        key = score.load_key(self.suite_dir / "QUALITY_KEY.jsonl")
        outputs = score.load_arm("paper", [Path(arm.split("=", 1)[1]) for arm in arms if arm.startswith("paper=")], key)
        self.assertEqual(outputs[short].missing, "stream-incomplete")
        self.assertEqual(outputs[rejected].missing, "rejected")

    def test_runs_of_other_prompts_or_duplicate_shards_are_refused(self):
        rows = {row["request_id"]: row for row in self.key}
        first = self.confirmatory[0]["request_id"]
        arms = self.write_arm("legacy", self.baseline_outputs(),
                              prompt_override={first: "sha256:" + "0" * 64})
        key = score.load_key(self.suite_dir / "QUALITY_KEY.jsonl")
        with self.assertRaisesRegex(score.QualityScoreError, "prompt or index"):
            score.load_arm("legacy", [Path(arm.split("=", 1)[1]) for arm in arms], key)
        run = self.root / "dup" / "run"
        fx.write_run(run, rows[first]["trace_name"], self.key, self.baseline_outputs())
        with self.assertRaisesRegex(score.QualityScoreError, "two runs"):
            score.load_arm("dup", [run, run], key)

    def test_stream_chunks_map_the_answer_to_its_token_prefix(self):
        path = self.root / "stream.raw"
        fx.write_stream(path, fx.answer_pieces("18", BUDGET), chunk=4)
        tokens, text, starts = score.read_stream(path)
        self.assertEqual(len(tokens), BUDGET)
        self.assertEqual(starts, [(0, 0), (4, len("Step one.\nFinal")), (8, len("Step one.\nFinal answer: 18\n"))])
        row = score.KeyRow("r", "t", 0, "qwen", "i", score.Decimal(18), 0, "sha256:x", BUDGET)
        output = score.Output(key=row, tokens=tokens, text=text, chunk_starts=starts,
                              extraction=score.extract_answer(text))
        self.assertEqual(output.extraction.to_json()["value"], "18")
        self.assertEqual(output.answer_token_end(), 8)

    def test_interim_is_blinded_and_bounded(self):
        arms = self.write_arm("legacy", self.baseline_outputs()) + self.write_arm("paper", self.paper_outputs())
        args = self.namespace(arms, ["paper"], shards=2)
        value = score.interim(args)
        self.assertEqual(value["discordance"]["paper"], {"pairs": 12, "discordant": 2, "discordance": 2 / 12})
        encoded = json.dumps(value)
        for leaked in ("accuracy", "gain", "loss", "difference", "correct"):
            self.assertNotIn(leaked, encoded)
        self.assertGreaterEqual(value["final_shards"], protocol.PLANNED_SHARDS)
        self.assertLessEqual(value["final_shards"], protocol.MAXIMUM_SHARDS)
        self.assertTrue(value["capped"])

    def test_command_line_writes_report_and_markdown(self):
        arms = self.write_arm("legacy", self.baseline_outputs()) + self.write_arm("paper", self.paper_outputs())
        out, md = self.root / "report.json", self.root / "report.md"
        argv = ["score", "score", "--suite", str(self.suite_dir / "SUITE.json"),
                "--key", str(self.suite_dir / "QUALITY_KEY.jsonl"), "--baseline", "legacy", "--treatment", "paper",
                "--shards", "2", "--bootstrap-replicates", "100", "--out", str(out), "--md", str(md)]
        for arm in arms:
            argv += ["--arm", arm]
        with mock.patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
            self.assertEqual(score.main(), 0)
        report = json.loads(out.read_text())
        self.assertEqual(report["schema"], score.REPORT_SCHEMA)
        self.assertIn("Conclusion (fixed sequence)", md.read_text())


class QualityPilotTests(unittest.TestCase):
    def test_pilot_picks_the_smallest_budget_holding_the_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite_dir = root / "pilot"
            per_role = protocol.PILOT_PER_MODEL
            fx.build_small_suite(suite_dir, shards=1, per_shard={"llama": per_role, "gemma": per_role, "qwen": per_role},
                                 output_tokens=protocol.PILOT_OUTPUT_TOKENS, items=3 * per_role, pilot=True)
            key = fx.key_rows(suite_dir)
            late = {"qwen": 1, "gemma": 2, "llama": 0}
            silent = {"qwen": 0, "gemma": 0, "llama": 2}
            outputs = {}
            for role in ("qwen", "gemma", "llama"):
                rows = sorted((row for row in key if row["role"] == role), key=lambda row: row["request_id"])
                for index, row in enumerate(rows):
                    if index < silent[role]:
                        value, position = None, 0
                    else:
                        value, position = row["reference"], 300 if index < late[role] else 100
                    outputs[row["request_id"]] = fx.answer_pieces(value, protocol.PILOT_OUTPUT_TOKENS,
                                                                  position=position)
            run = root / "run"
            fx.write_run(run, key[0]["trace_name"], key, outputs)
            args = argparse.Namespace(suite=suite_dir / "SUITE.json", key=suite_dir / "QUALITY_KEY.jsonl",
                                      arm=[f"legacy={run}"])
            value = score.pilot(args)
            roles = value["roles"]
            self.assertEqual(roles["qwen"]["allowed_misses"], 1)
            self.assertEqual(roles["qwen"]["misses_by_budget"], {"256": 1, "320": 0, "384": 0, "512": 0})
            self.assertEqual(roles["qwen"]["chosen_output_tokens"], 256)
            self.assertEqual(roles["gemma"]["chosen_output_tokens"], 320)
            self.assertEqual(roles["llama"]["chosen_output_tokens"], 512)
            self.assertTrue(roles["llama"]["capped"])
            self.assertFalse(roles["gemma"]["capped"])
            self.assertEqual(roles["qwen"]["answer_token_end"]["max"], 307)
            self.assertEqual(value["builder_arguments"],
                             "--output-tokens qwen=256 --output-tokens gemma=320 --output-tokens llama=512")
            with self.assertRaisesRegex(score.QualityScoreError, "pilot"):
                score.score(argparse.Namespace(suite=suite_dir / "SUITE.json", key=suite_dir / "QUALITY_KEY.jsonl",
                                               arm=[f"legacy={run}"], baseline="legacy", treatment=["legacy"],
                                               shards=1, strict_arm=[], aa=None, bootstrap_replicates=10))


if __name__ == "__main__":
    unittest.main()
