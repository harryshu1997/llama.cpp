"""WS6 generality traces: later windows / shifted scan grids in the builder, and the variant tool.

The end-to-end cases run the real builder against a tiny synthetic BurstGPT CSV and a fake token codec
(one token per character, the chat models' BOS rules), so no tokenizer model or desktop is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest

from research_dev.scheduler.campaigns.burstgpt.build_realistic_trace import (
    inventory_from_manifest, select_window,
)
from research_dev.scheduler.campaigns.burstgpt.tools import generality_traces as g
from research_dev.scheduler.configuration.campaign import CampaignManifest

FROZEN_BASE = g.PAPER_CONFIG / "trace" / g.BASE_NAME
FAKE_CODEC = """#!{python}
import json, sys
args = sys.argv[1:]
model, sha = args[args.index("--model") + 1].rsplit("/", 1)[-1], args[args.index("--model-sha256") + 1]
bos = [128000] if "llama" in model else [2] if "gemma" in model else []
for line in sys.stdin:
    request = json.loads(line)
    response = {{"schema": "layersplit-token-codec-response-v1", "request_id": request["request_id"],
                "op": request["op"], "model_sha256": sha}}
    if request["op"] == "tokenize":
        response["tokens"] = bos + [ord(c) for c in request["text"]]
    else:
        response["text"] = "".join(chr(t) for t in request["tokens"] if t < 0x110000)
    sys.stdout.write(json.dumps(response) + "\\n")
    sys.stdout.flush()
"""
# small-trace rules: 600 s windows of 4-6 requests, at least one of each source model
TEST_RULES = ("--small-model-share", "0.5", "--duration-s", "600", "--min-requests", "4", "--max-requests", "6",
              "--prompt-cap", "300", "--output-cap", "100", "--min-requests-per-model", "1")


def rows(windows: int, per_window: int = 5, duration: float = 600.0) -> list[dict]:
    result = []
    for window in range(windows):
        for index in range(per_window):
            result.append({"t": window * duration + index * 60.0, "model": "GPT-4" if index % 3 == 0 else "ChatGPT",
                           "input": 100 + 10 * index, "output": 20 + window + index})
    return result


class SelectWindowRankTests(unittest.TestCase):
    RULES = dict(duration_s=600, min_requests=4, max_requests=6, start_offset_s=None, min_input=1, min_output=1)

    def test_default_is_the_first_window_without_new_keys(self) -> None:
        window, info = select_window(rows(4), **self.RULES)
        self.assertEqual(info["window_start_source_s"], 0.0)
        self.assertNotIn("window_rank", info)
        self.assertNotIn("scan_phase_s", info)
        self.assertEqual(len(window), 5)

    def test_later_ranks_are_the_next_qualifying_windows(self) -> None:
        data = rows(4) + [{"t": 5000.0, "model": "ChatGPT", "input": 100, "output": 50}]  # scan to 4800 s
        data[5:10] = data[5:6]  # window 1 keeps one row: it no longer qualifies
        _, second = select_window(data, **self.RULES, window_rank=2)
        self.assertEqual(second["window_start_source_s"], 1200.0)
        self.assertEqual(second["scanned_windows"], 8)
        self.assertEqual(second["window_rank"], {"rank": 2, "earlier_qualifying_starts_source_s": [0.0]})
        _, third = select_window(data, **self.RULES, window_rank=3)
        self.assertEqual(third["window_rank"]["earlier_qualifying_starts_source_s"], [0.0, 1200.0])
        with self.assertRaisesRegex(SystemExit, "only 3 qualifying windows"):
            select_window(data, **self.RULES, window_rank=4)
        with self.assertRaises(SystemExit):
            select_window(data, **{**self.RULES, "start_offset_s": 0.0}, window_rank=2)
        with self.assertRaises(SystemExit):
            select_window(data, **self.RULES, window_rank=0)

    def test_scan_phase_shifts_the_grid(self) -> None:
        # rows every 60 s from 0 to 2340 s; with the grid shifted by 300 s the first window is [300, 900)
        data = [{"t": 60.0 * k, "model": "ChatGPT", "input": 100, "output": 50} for k in range(40)]
        window, info = select_window(data, **{**self.RULES, "min_requests": 10, "max_requests": 10},
                                     scan_phase_s=300.0)
        self.assertEqual((info["window_start_source_s"], info["scan_phase_s"], info["scanned_windows"]),
                         (300.0, 300.0, 3))
        self.assertEqual((window[0]["t"], window[-1]["t"]), (300.0, 840.0))
        for phase in (-1.0, 600.0):
            with self.subTest(phase=phase), self.assertRaises(SystemExit):
                select_window(data, **self.RULES, scan_phase_s=phase)


class InventoryFromManifestTests(unittest.TestCase):
    def test_copies_the_frozen_inventory_and_refuses_bad_ones(self) -> None:
        manifest = json.loads((FROZEN_BASE / "TRACE_MANIFEST.json").read_text())
        self.assertEqual(inventory_from_manifest(FROZEN_BASE / "TRACE_MANIFEST.json"), manifest["model_inventory"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "M.json"
            bad_rows = (
                {"model_inventory": {}},
                {"model_inventory": {"gemma-4-12b-it-q8_0": manifest["model_inventory"]["llama-3.2-1b-instruct-q4_0"]}},
                {"model_inventory": {"x": {**manifest["model_inventory"]["llama-3.2-1b-instruct-q4_0"],
                                           "artifact_sha256": "ab"}}},
            )
            for value in bad_rows:
                path.write_text(json.dumps(value))
                with self.subTest(value=value), self.assertRaises(SystemExit):
                    inventory_from_manifest(path)


class VariantToolTests(unittest.TestCase):
    def test_variants_change_exactly_one_rule(self) -> None:
        by_name = {row.name.removeprefix(g.BASE_NAME + "_"): row for row in g.VARIANTS}
        self.assertEqual(by_name["w2"].builder_flags(), ("--window-rank", "2"))
        self.assertEqual(by_name["w4"].builder_flags(), ("--scan-phase-s", "900.0"))
        self.assertEqual(by_name["w5"].builder_flags(), ("--window-rank", "2", "--scan-phase-s", "900.0"))
        self.assertEqual(by_name["d2x"].builder_flags(), ("--arrival-scale", "0.5"))
        self.assertEqual(by_name["d4x"].builder_flags(), ("--arrival-scale", "0.25"))
        self.assertEqual(by_name["d0p5x"].builder_flags(), ("--arrival-scale", "2.0"))
        self.assertEqual(g.BASE.builder_flags(), ())
        manifest = json.loads((FROZEN_BASE / "TRACE_MANIFEST.json").read_text())
        derivation = manifest["derivation"]
        rules = dict(zip(g.BASE_RULES[::2], g.BASE_RULES[1::2]))
        window = derivation["window"]
        self.assertEqual((float(rules["--small-model-share"]), int(rules["--prompt-cap"]), int(rules["--output-cap"]),
                          float(rules["--duration-s"]), int(rules["--max-output-tokens"]),
                          int(rules["--long-tail-threshold"]), float(rules["--long-tail-tolerance"])),
                         (derivation["small_model_share"], derivation["prompt_cap"], derivation["output_cap"],
                          window["window_duration_s"], window["max_output_tokens"],
                          window["long_tail"]["log"]["threshold_tokens"], window["long_tail"]["tolerance"]))

    def test_relocation_rewrites_only_the_recorded_paths(self) -> None:
        manifest = json.loads((FROZEN_BASE / "TRACE_MANIFEST.json").read_text())
        moved = g.relocate_manifest(manifest, "/somewhere/else", "/x/burstgpt_3.csv")
        self.assertEqual(moved["base_trace"]["path"], "/somewhere/else/REQUESTS_SEMANTIC_SOURCE.jsonl")
        self.assertEqual(moved["overlay_trace"]["path"], "/somewhere/else/REQUESTS_OVERLAY.jsonl")
        self.assertEqual(moved["derivation"]["burstgpt_csv"], "/x/burstgpt_3.csv")
        back = g.relocate_manifest(moved, g.DESKTOP_ROOT + "/" + g.BASE_NAME, g.DESKTOP_CSV)
        self.assertEqual(g.canonical(back), (FROZEN_BASE / "TRACE_MANIFEST.json").read_bytes())

    def test_template_changes_only_the_trace_paths(self) -> None:
        variant = g.VARIANT_BY_NAME[g.BASE_NAME + "_d2x"]
        source = g.PAPER_CONFIG / "template"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / variant.template_name
            sums = g.build_template(source, destination, variant)
            self.assertEqual(set(sums), set(g.TEMPLATE_FILES))
            campaign = json.loads((destination / "campaign.json").read_text())
            self.assertEqual(campaign["trace"]["replay_schedule_path"],
                             f"/mnt/storage/burstgpt-source/{variant.name}/burstgpt_{variant.name}.json")
            parsed = CampaignManifest.from_json(campaign, destination)
            self.assertEqual(str(parsed.trace.trace_manifest_path),
                             f"/mnt/storage/burstgpt-source/{variant.name}/TRACE_MANIFEST.json")
            changes = (destination / "CHANGES.txt").read_text()
            self.assertIn(f"/mnt/storage/burstgpt-source/{variant.name}/TRACE_MANIFEST.json", changes)
            self.assertTrue(changes.startswith((source / "CHANGES.txt").read_text().split("\n")[0]))
            campaign["selection_mode"] = "desktop-baseline"
            (destination / "campaign.json").write_text(json.dumps(campaign))
            with self.assertRaisesRegex(g.GeneralityError, "beyond the trace paths"):
                g.check_template(source, destination, variant)

    def test_thread_template_sets_only_the_assisted_models_threads(self) -> None:
        from research_dev.scheduler.configuration.models import ModelsManifest

        source = g.PAPER_CONFIG / "template"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "template-eval2-s2-t4p"
            g.thread_template(source, destination, 4, "0,2,4,6")
            models = json.loads((destination / "models.json").read_text())
            by_kind = {row["model_id"]: row["runtime_parameters"] for row in models["models"]}
            for model_id in ("qwen3-14b-q4km-dequant-f16", "gemma-4-12b-q40-dequant-f16"):
                self.assertEqual({key: by_kind[model_id][key] for key in g.THREAD_KEYS},
                                 {"threads": 4, "threads_batch": 4, "cpu_affinity": "0,2,4,6"})
            self.assertEqual(by_kind["llama-3.2-1b-instruct-q4_0"], {})
            parsed = ModelsManifest.from_json(models, destination)
            self.assertEqual(parsed.models[0].runtime_parameters["threads"], 4)
            self.assertIn("threads=4 threads_batch=4 cpu_affinity=0,2,4,6", (destination / "CHANGES.txt").read_text())
            with self.assertRaisesRegex(g.GeneralityError, "already sets threads"):
                g.thread_template(destination, Path(directory) / "again", 8)
            with self.assertRaisesRegex(g.GeneralityError, "affinity"):
                g.thread_template(source, Path(directory) / "bad", 8, "0;2")

    def test_disjoint_windows(self) -> None:
        g.check_disjoint({"a": (0.0, 1800.0), "b": (1800.0, 3600.0), "c": (9000.0, 10800.0)})
        with self.assertRaisesRegex(g.GeneralityError, "overlap"):
            g.check_disjoint({"a": (0.0, 1800.0), "b": (900.0, 2700.0)})

    def test_frozen_base_loads_like_the_campaign(self) -> None:
        trace = g.load_trace(FROZEN_BASE, g.BASE.trace_name)
        summary = g.trace_summary(trace)
        self.assertEqual((summary["requests"], summary["output_tokens"], summary["replay_span_s"]), (14, 3604, 1675.0))
        self.assertEqual(summary["source_models"], {"ChatGPT": 8, "GPT-4": 6})


class BuildEndToEndTests(unittest.TestCase):
    """The real builder, a fake codec and a synthetic CSV: window rank, density and loader identity."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        codec = root / "fake-codec"
        codec.write_text(FAKE_CODEC.format(python=sys.executable))
        codec.chmod(codec.stat().st_mode | stat.S_IXUSR)
        cls.tools = {"codec": codec, "library_dir": root}
        for key, name in (("qwen", "qwen.gguf"), ("gemma", "gemma.gguf"), ("llama", "llama.gguf")):
            (root / name).write_bytes(name.encode())
            cls.tools[key] = root / name
        cls.csv = root / "burstgpt.csv"
        lines = ["Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type"]
        for row in rows(3) + [{"t": 5000.0, "model": "ChatGPT", "input": 100, "output": 20}]:
            lines.append(f"{row['t']:.0f},{row['model']},{row['input']},{row['output']},0,Conversation log")
        lines.append("10,ChatGPT,100,20,0,API log")
        cls.csv.write_text("\n".join(lines) + "\n")
        cls.out = root / "out"
        cls.base = g.Variant("syn", "base", "synthetic base")
        cls.second = g.Variant("syn_w2", "window", "second window", window_rank=2)
        cls.dense = g.Variant("syn_d2x", "density", "denser", arrival_scale=0.5)
        cls.built = {variant.name: g.build_variant(variant, out_dir=cls.out, csv=cls.csv, tools=cls.tools,
                                                   inventory_from=FROZEN_BASE / "TRACE_MANIFEST.json",
                                                   rules=TEST_RULES)
                     for variant in (cls.base, cls.second, cls.dense)}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_built_traces_load_and_carry_desktop_paths_and_sums(self) -> None:
        for variant in (self.base, self.second, self.dense):
            directory = self.built[variant.name]
            trace = g.load_trace(directory, variant.trace_name)
            manifest = trace["manifest"]
            self.assertEqual(manifest["base_trace"]["path"],
                             f"/mnt/storage/burstgpt-source/{variant.name}/REQUESTS_SEMANTIC_SOURCE.jsonl")
            self.assertEqual(manifest["derivation"]["burstgpt_csv"], g.DESKTOP_CSV)
            sums = dict(reversed(line.split("  ", 1)) for line in (directory / "SHA256SUMS").read_text().splitlines())
            self.assertEqual(set(sums), set(g.trace_file_names(variant.trace_name)))
            record = json.loads((directory / "VARIANT.json").read_text())
            self.assertEqual(record["desktop_directory"], f"/mnt/storage/burstgpt-source/{variant.name}")
            self.assertEqual(record["builder_command"][:3], ["python3", "-m", g.BUILDER_MODULE])
            self.assertIn(f"/mnt/storage/burstgpt-source/{variant.name}", record["builder_command"])
            self.assertEqual(len(trace["overlay"]), 2)

    def test_window_rank_takes_a_later_disjoint_window(self) -> None:
        base = g.load_trace(self.built["syn"], self.base.trace_name)["manifest"]["derivation"]["window"]
        second = g.load_trace(self.built["syn_w2"], self.second.trace_name)["manifest"]["derivation"]["window"]
        self.assertEqual((base["window_start_source_s"], second["window_start_source_s"]), (0.0, 600.0))
        self.assertEqual(second["window_rank"], {"rank": 2, "earlier_qualifying_starts_source_s": [0.0]})
        g.check_disjoint({"syn": (0.0, 600.0), "syn_w2": (600.0, 1200.0)})

    def test_density_variant_is_the_base_with_scaled_arrivals(self) -> None:
        base = g.load_trace(self.built["syn"], self.base.trace_name)
        dense = g.load_trace(self.built["syn_d2x"], self.dense.trace_name)
        g.check_density(base, dense, 0.5, self.base.trace_name, self.dense.trace_name)
        self.assertEqual(dense["replay"]["replay_span_us"] * 2, base["replay"]["replay_span_us"])
        with self.assertRaisesRegex(g.GeneralityError, "density arrival"):
            g.check_density(base, dense, 0.25, self.base.trace_name, self.dense.trace_name)
        tampered = {**dense, "large": [dict(dense["large"][0], output_tokens=1), *dense["large"][1:]]}
        with self.assertRaisesRegex(g.GeneralityError, "beyond arrival and id"):
            g.check_density(base, tampered, 0.5, self.base.trace_name, self.dense.trace_name)

    def test_copy_script_names_every_destination(self) -> None:
        script = g.copy_script([self.base, self.dense])
        self.assertIn("scp trace/syn_d2x/* $H:/mnt/storage/burstgpt-source/syn_d2x/", script)
        self.assertIn("sha256sum -c SHA256SUMS", script)
        self.assertIn(f"scp -r templates/{self.dense.template_name} $H:{g.DESKTOP_TEMPLATE_ROOT}/", script)


if __name__ == "__main__":
    unittest.main()
