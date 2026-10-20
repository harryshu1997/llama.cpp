#!/usr/bin/env python3
"""WS17 quantized regime: per-model matrix precision, calibration priors, approximate-numerics receipts and the
offline preparation tool. Every path is opt-in: campaigns without the new keys are byte-identical."""

from __future__ import annotations

from dataclasses import replace
import http.server
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parents[2]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from research_dev.scheduler._internal import approximate_numerics as an  # noqa: E402
from research_dev.scheduler._internal.matrix_precision import (  # noqa: E402
    CalibrationPrior, MatrixPrecisionError, MatrixPrecisionSpec, evaluate,
)
from research_dev.scheduler._internal.model_manifest import ModelManifest  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt import catalog, launch  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import approximate_numerics_gate as gate  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import quantized_regime as regime  # noqa: E402
from research_dev.scheduler.config import load_scheduler_configuration  # noqa: E402
from research_dev.scheduler.configuration.models import CampaignModelConfiguration, ModelsManifest  # noqa: E402

DATA = TESTS_DIR.parent / "campaigns/burstgpt/data"
PAPER = TESTS_DIR.parent / "campaigns/burstgpt/paper_config_v1"
PRIOR = {"schema": "ws17-calibration-prior-v1", "desktop_service_scale_ppm": 500_000,
         "assisted_service_scale_ppm": 1_500_000, "receipt_sha256": "sha256:" + "a" * 64, "phone_columns": "model"}


def tensor(name, quantization, shape=(5120, 5120)):
    return SimpleNamespace(tensor_id=name, quantization=quantization, shape=shape)


class MatrixPrecisionTests(unittest.TestCase):
    def test_spec_validation(self):
        spec = MatrixPrecisionSpec.from_json({"allowed": ["Q4_0", "Q6_K"], "ffn": "Q4_0"})
        self.assertEqual(spec.to_json(), {"allowed": ["Q4_0", "Q6_K"], "ffn": "Q4_0"})
        for bad in ({"allowed": []}, {"allowed": ["Q4_0"], "ffn": "Q8_0"}, {"allowed": ["Q9_9"]},
                    {"allowed": ["Q4_0", "Q4_0"]}, {"allowed": ["Q4_0"], "extra": 1}, ["Q4_0"]):
            with self.assertRaises(MatrixPrecisionError, msg=str(bad)):
                MatrixPrecisionSpec.from_json(bad)

    def test_plain_q4_0_passes_and_imatrix_q4_1_down_fails_the_ffn_rule(self):
        spec = MatrixPrecisionSpec.from_json({"allowed": ["Q4_0", "Q4_1", "Q6_K"], "ffn": "Q4_0"})
        rows = [tensor("blk.0.ffn_gate.weight", "Q4_0"), tensor("blk.0.ffn_up.weight", "Q4_0"),
                tensor("blk.0.ffn_down.weight", "Q4_0"), tensor("output.weight", "Q6_K"),
                tensor("blk.0.attn_norm.weight", "F32", (5120,))]
        self.assertTrue(evaluate(rows, spec)[0])
        rows[2] = tensor("blk.0.ffn_down.weight", "Q4_1")
        ok, detail = evaluate(rows, spec)
        self.assertFalse(ok)
        self.assertIn("blk.0.ffn_down.weight", detail)
        self.assertFalse(evaluate([tensor("blk.0.ffn_up.weight", "F32")], MatrixPrecisionSpec(("Q4_0",)))[0])
        self.assertTrue(evaluate([tensor("blk.0.attn_q.weight", "Q4_K"), tensor("blk.0.ffn_down.weight", "Q6_K")],
                                 MatrixPrecisionSpec(("Q4_K", "Q6_K")))[0])

    def test_models_round_trip_is_byte_identical_without_the_keys(self):
        raw = json.loads((PAPER / "template/models.json").read_text())
        for row in raw["models"]:
            config = CampaignModelConfiguration.from_json(row, Path("/"))
            self.assertNotIn("matrix_precision", config.to_json())
            self.assertNotIn("calibration_prior", config.to_json())
            self.assertIsNone(config.matrix_precision)
        hot = dict(raw["models"][0], matrix_precision={"allowed": ["Q4_0", "Q6_K"], "ffn": "Q4_0"},
                   calibration_prior=PRIOR)
        config = CampaignModelConfiguration.from_json(hot, Path("/"))
        self.assertEqual(config.to_json()["matrix_precision"], hot["matrix_precision"])
        self.assertEqual(config.to_json()["calibration_prior"], PRIOR)

    def test_preflight_command_carries_the_spec_only_when_declared(self):
        from test_two_phone_helpers import CampaignConfigurationTests, legacy_rig_json

        case = CampaignConfigurationTests()
        case.setUp()
        self.addCleanup(case.tearDown)
        path = case._write(legacy_rig_json(), None)
        command_of = lambda configuration: launch.preflight_command(  # noqa: E731
            configuration, catalog_path=case.root / "c.json", normal_usb_receipt_path=case.root / "u.json",
            output_path=case.root / "o.json")
        plain = command_of(load_scheduler_configuration(path, environ={}))
        self.assertNotIn("--qwen-matrix-precision", plain)
        self.assertNotIn("--gemma-matrix-precision", plain)
        models = json.loads((case.root / "models.json").read_text())
        models["models"][0]["matrix_precision"] = {"allowed": ["Q4_0", "Q6_K"], "ffn": "Q4_0"}
        (case.root / "models.json").write_text(json.dumps(models))
        command = command_of(load_scheduler_configuration(path, environ={}))
        self.assertEqual(json.loads(command[command.index("--qwen-matrix-precision") + 1]),
                         {"allowed": ["Q4_0", "Q6_K"], "ffn": "Q4_0"})
        self.assertNotIn("--gemma-matrix-precision", command)
        self.assertEqual(command[:len(plain)], plain)

    def test_preflight_model_checks_default_to_f16(self):
        from research_dev.scheduler.campaigns.burstgpt import preflight

        manifest = SimpleNamespace(tensors=(tensor("blk.0.ffn_up.weight", "Q4_0"),))
        f16 = SimpleNamespace(tensors=(tensor("blk.0.ffn_up.weight", "F16"),))
        self.assertEqual(preflight._precision_check("x", f16, "F16").status, "PASS")
        self.assertEqual(preflight._precision_check("x", manifest, "F16").status, "BLOCKED")
        self.assertEqual(preflight._spec_precision_check(
            "x", manifest, MatrixPrecisionSpec(("Q4_0",), "Q4_0")).status, "PASS")


class CalibrationPriorTests(unittest.TestCase):
    def setUp(self):
        self.calibration = {arm: [catalog.physical_result(DATA / "calibration", name) for name in names]
                            for arm, names in (("control", ("control-r1b", "control-r2")),
                                               ("op15", ("op15-r1", "op15-r2")))}
        raw = json.loads((PAPER / "template/models.json").read_text())
        self.configs = {row["model_key"]: CampaignModelConfiguration.from_json(row, Path("/"))
                        for row in raw["models"] if row["kind"] == "assisted"}
        self.manifests = {"hot": ModelManifest.from_json(json.loads((DATA / "QWEN_MANIFEST.json").read_text())),
                          "cold": ModelManifest.from_json(json.loads((DATA / "GEMMA_MANIFEST.json").read_text()))}

    def profiles(self, configs, manifests):
        rows = catalog.route_profiles(tuple(configs), manifests, self.calibration,
                                      ("desktop-cpu", "desktop-cuda", "op15-phone"), "control", "op15")
        return {row.selector_id: row for row in rows}

    def test_prior_validation(self):
        self.assertEqual(CalibrationPrior.from_json(PRIOR).to_json(), PRIOR)
        for bad in ({**PRIOR, "desktop_service_scale_ppm": 5}, {**PRIOR, "receipt_sha256": "x"},
                    {**PRIOR, "phone_columns": "maybe"}, {**PRIOR, "schema": "other"}, {**PRIOR, "extra": 1}):
            with self.assertRaises(MatrixPrecisionError):
                CalibrationPrior.from_json(bad)

    def test_route_profiles_scale_per_route_and_rescale_columns(self):
        base = self.profiles(self.configs.values(), self.manifests)
        prior = CalibrationPrior.from_json(PRIOR)
        hot = replace(self.configs["hot"], calibration_prior=prior)
        wide = replace(self.manifests["hot"], feed_forward_length=25600, artifact_sha256="sha256:" + "b" * 64)
        with self.assertRaises(Exception):          # the recorded 17,408 columns are not a full 25,600 replacement
            self.profiles([self.configs["hot"], self.configs["cold"]], {**self.manifests, "hot": wide})
        scaled = self.profiles([hot, self.configs["cold"]], {**self.manifests, "hot": wide})
        self.assertEqual(set(scaled), set(base))
        for selector, row in scaled.items():
            original = base[selector]
            if not selector.startswith("physical:hot:"):
                self.assertEqual(row, original)
                continue
            factor = 0.5 if selector.startswith("physical:hot:control:") else 1.5
            self.assertEqual(row.artifact_sha256, "sha256:" + "b" * 64)
            self.assertAlmostEqual(row.service_output_token_us, original.service_output_token_us * factor, delta=1)
            self.assertAlmostEqual(row.service_fixed_us, original.service_fixed_us * factor, delta=1)
            self.assertIn(prior.receipt_sha256, row.evidence_ids)
            self.assertEqual(row.maturity, "SHADOW")


class ApproximateNumericsTests(unittest.TestCase):
    @staticmethod
    def path(tokens, tops, role="reference"):
        prompts = [{"prompt_id": "p0", "prompt_tokens": [1, 2],
                    "positions": [{"token": t, "top": top} for t, top in zip(tokens, tops)]}]
        value = {"schema": an.PATH_SCHEMA, "role": role, "prompts": prompts}
        if role == "reference":
            value["path_sha256"] = gate.path_sha256(prompts)
        return value

    def test_row_parsing_and_screen(self):
        lines = ["S41SERVERFFNROW stage=local call=1 layer=3 local_rel_l2=-1.000e+00",
                 "S41SERVERFFNROW stage=returned call=1 layer=3 ubatch_row=0 local_rel_l2=4.000e-03 local_max_abs=1e-2",
                 "S41SERVERFFNROW stage=returned call=2 layer=4 ubatch_row=0 local_rel_l2=inf local_max_abs=1",
                 "S41SERVERFFNROW stage=returned call=3 layer=4 ubatch_row=0 local_rel_l2=-1.000e+00 x=1",
                 "noise"]
        rows = an.parse_row_diagnostics(lines)
        self.assertEqual([(row["layer"], row["rel_l2"]) for row in rows], [(3, 4e-3), (4, float("inf"))])
        policy = an.ApproximateNumericsPolicy(min_row_samples=2)
        self.assertEqual(an.row_screen(rows, policy)["status"], "FAIL")
        good = [{"layer": i % 4, "rel_l2": 3e-3} for i in range(300)]
        self.assertEqual(an.row_screen(good, an.ApproximateNumericsPolicy())["status"], "PASS")
        high = good[:290] + [{"layer": 1, "rel_l2": 0.5}] * 10
        self.assertIn("p99", " ".join(an.row_screen(high, an.ApproximateNumericsPolicy())["reasons"]))

    def test_teacher_forced_bound_uses_the_aa_band(self):
        n = 300
        reference = self.path([7] * n, [[[7, -0.1], [9, -2.5]]] * n)
        flips = lambda count: [[[9, -0.4], [7, -0.5]]] * count + [[[7, -0.1], [9, -2.5]]] * (n - count)  # noqa: E731
        candidate = self.path([0] * n, flips(4), role="forced")
        aa = self.path([0] * n, flips(3), role="forced")
        for value in (candidate, aa):
            value["forced_reference_sha256"] = reference["path_sha256"]
        rows = [{"layer": 0, "rel_l2": 3e-3}] * 300
        receipt = an.evaluate(reference=reference, candidate=candidate, aa_candidate=aa, rows=rows)
        self.assertEqual(receipt["status"], "PASS")
        self.assertAlmostEqual(receipt["teacher_forced"]["bound"], max(0.01, 2 * 3 / n + 0.005))
        self.assertEqual(receipt["teacher_forced"]["phone"]["near_tie_disagreements"], 0)
        worse = self.path([0] * n, flips(12), role="forced")
        worse["forced_reference_sha256"] = reference["path_sha256"]
        self.assertEqual(an.evaluate(reference=reference, candidate=worse, aa_candidate=aa, rows=rows)["status"], "FAIL")
        other = dict(candidate, forced_reference_sha256="sha256:" + "0" * 64)
        with self.assertRaises(an.ApproximateNumericsError):
            an.evaluate(reference=reference, candidate=other, aa_candidate=aa, rows=rows)
        policy = an.ApproximateNumericsPolicy.from_json({"policy": an.POLICY_ID, "min_positions": 1000})
        self.assertEqual(an.evaluate(reference=reference, candidate=candidate, aa_candidate=aa, rows=rows,
                                     policy=policy)["status"], "FAIL")
        with self.assertRaises(an.ApproximateNumericsError):
            an.ApproximateNumericsPolicy.from_json({"absolute_floor": 1})       # int, not float

    def test_client_against_a_fake_server(self):
        calls = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(body)
                count = body["n_predict"]
                start = len(body["prompt"])
                probs = [{"id": 100 + start + i, "logprob": -0.1,
                          "top_logprobs": [{"id": 100 + start + i, "logprob": -0.1}, {"id": 1, "logprob": -3.0}]}
                         for i in range(count)]
                data = json.dumps({"completion_probabilities": probs}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        endpoint = f"http://127.0.0.1:{server.server_port}"
        prompts = [{"prompt_id": "p0", "prompt_tokens": [5, 6, 7]}]
        reference = gate.reference_path(endpoint, prompts, 4, 10)
        self.assertEqual([p["token"] for p in reference["prompts"][0]["positions"]], [103, 104, 105, 106])
        forced = gate.forced_path(endpoint, reference, 10)
        self.assertEqual([p["token"] for p in forced["prompts"][0]["positions"]], [103, 104, 105, 106])
        self.assertEqual(calls[-1]["prompt"], [5, 6, 7, 103, 104, 105])
        self.assertTrue(all(c["ignore_eos"] and c["temperature"] == 0.0 and c["n_probs"] == 2 for c in calls))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("ref", reference), ("cand", forced), ("aa", forced),
                                ("rows", {"rows": [{"layer": 0, "rel_l2": 1e-3}] * 300})):
                (root / (name + ".json")).write_text(json.dumps(value))
            (root / "policy.json").write_text(json.dumps({"min_positions": 4}))
            rc = gate.main(["evaluate", "--reference", str(root / "ref.json"), "--candidate", str(root / "cand.json"),
                            "--aa", str(root / "aa.json"), "--rows", str(root / "rows.json"),
                            "--policy", str(root / "policy.json"), "--identity", "artifact=sha256:x",
                            "--out", str(root / "receipt.json")])
            self.assertEqual(rc, 0)
            receipt = json.loads((root / "receipt.json").read_text())
            self.assertEqual((receipt["status"], receipt["identity"]["artifact"]), ("PASS", "sha256:x"))


def write_gguf(path: Path, *, tokens: list[str], bos: int = 1, ffn_type: str = "Q4_0", blocks: int = 2) -> None:
    sys.path.insert(0, str(REPO / "gguf-py"))
    import gguf
    import numpy as np

    writer = gguf.GGUFWriter(str(path), "qwen3")
    writer.add_block_count(blocks)
    writer.add_embedding_length(64)
    writer.add_feed_forward_length(128)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_key_length(16)
    writer.add_value_length(16)
    writer.add_context_length(4096)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_types([1] * len(tokens))
    writer.add_token_merges(["a b"])
    writer.add_bos_token_id(bos)
    writer.add_add_bos_token(False)
    q_type = getattr(gguf.GGMLQuantizationType, ffn_type)
    block, size = gguf.GGML_QUANT_SIZES[q_type]

    def add(name, rows, columns, quantization=gguf.GGMLQuantizationType.F16):
        if quantization == gguf.GGMLQuantizationType.F16:
            writer.add_tensor(name, np.zeros((rows, columns), dtype=np.float16))
        else:
            qb, qs = gguf.GGML_QUANT_SIZES[quantization]
            writer.add_tensor(name, np.zeros((rows, columns // qb * qs), dtype=np.uint8), raw_dtype=quantization)

    for i in range(blocks):
        add(f"blk.{i}.attn_q.weight", 64, 64)
        add(f"blk.{i}.attn_k.weight", 32, 64)
        add(f"blk.{i}.attn_v.weight", 32, 64)
        add(f"blk.{i}.attn_output.weight", 64, 64)
        add(f"blk.{i}.ffn_gate.weight", 128, 64, q_type)
        add(f"blk.{i}.ffn_up.weight", 128, 64, q_type)
        add(f"blk.{i}.ffn_down.weight", 64, 128, q_type)
        writer.add_tensor(f"blk.{i}.attn_norm.weight", np.zeros(64, dtype=np.float32))
    add("output.weight", len(tokens), 64)
    add("token_embd.weight", len(tokens), 64)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class RegimeToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_tokenizer_check(self):
        tokens = [f"t{i}" for i in range(32)]
        write_gguf(self.root / "a.gguf", tokens=tokens)
        write_gguf(self.root / "b.gguf", tokens=tokens, bos=5, ffn_type="Q8_0")
        write_gguf(self.root / "c.gguf", tokens=tokens[:-1] + ["other"])
        same = regime.tokenizer_check(self.root / "a.gguf", self.root / "b.gguf")
        self.assertEqual((same["status"], same["special_ids_differ"]), ("PASS", ["tokenizer.ggml.bos_token_id"]))
        self.assertEqual(regime.tokenizer_check(self.root / "a.gguf", self.root / "c.gguf")["differs"],
                         ["tokenizer.ggml.tokens"])

    def test_vram_fit(self):
        write_gguf(self.root / "a.gguf", tokens=[f"t{i}" for i in range(32)], blocks=4)
        fit = regime.vram_fit(self.root / "a.gguf", context_size=1024, parallel=2, ubatch=64, gpu_first_layer=None,
                              free_vram_bytes=10 ** 9, reserve_bytes=0, workspace_bytes=0, cuda_context_bytes=0)
        self.assertEqual((fit["fits"], fit["gpu_first_layer"], fit["gpu_layers"]), (True, 0, 4))
        kv = 4 * (32 + 32) * 2 * 1024
        self.assertEqual(fit["footprint_bytes"]["kv"], kv)
        tight = regime.vram_fit(self.root / "a.gguf", context_size=1024, parallel=2, ubatch=64, gpu_first_layer=None,
                                free_vram_bytes=fit["footprint_bytes"]["total"] - 1, reserve_bytes=0,
                                workspace_bytes=0, cuda_context_bytes=0)
        self.assertEqual(tight["gpu_first_layer"], 1)
        none = regime.vram_fit(self.root / "a.gguf", context_size=1024, parallel=2, ubatch=64, gpu_first_layer=None,
                               free_vram_bytes=1, reserve_bytes=0, workspace_bytes=0, cuda_context_bytes=0)
        self.assertFalse(none["fits"])

    def test_trace_variant_keeps_the_trace_identity(self):
        source = PAPER / "trace/longtail_eval_v2"
        specs = ["qwen3-14b-q4km-gguf=@" + "1" * 64 + ":9001752960:Qwen3-14B-Q4_K_M.gguf",
                 "gemma-4-12b-q40-gguf=@" + "2" * 64 + ":6975878176:gemma-4-12B-it-Q4_0.gguf",
                 "llama-3.2-1b-instruct-q4_0=@4b90b1d7ae7324676194755a6dfce11cb6e457982c4c01a1db2857be1ed064ad"
                 ":770928288:Llama-3.2-1B-Instruct-Q4_0.gguf"]
        out = regime.trace_variant(source, self.root / "q4", specs)
        manifest = json.loads((self.root / "q4/TRACE_MANIFEST.json").read_text())
        self.assertEqual(sorted(manifest["model_inventory"]), sorted(spec.split("=")[0] for spec in specs))
        self.assertEqual((self.root / "q4/REQUESTS_SEMANTIC_SOURCE.jsonl").read_bytes(),
                         (source / "REQUESTS_SEMANTIC_SOURCE.jsonl").read_bytes())
        self.assertIn("source_manifest_sha256", manifest["ws17_variant"])
        self.assertEqual(out["model_inventory"], manifest["model_inventory"])
        with self.assertRaises(regime.RegimeError):     # a trace role id is not an execution model id
            regime.trace_variant(source, self.root / "bad", ["qwen3-14b-q4_k_m=@" + "1" * 64 + ":1:x"] + specs[1:])
        with self.assertRaises(regime.RegimeError):     # the overlay model must stay
            regime.trace_variant(source, self.root / "bad2", specs[:2] + ["other=@" + "3" * 64 + ":1:x"])

    def test_plans_seed_and_template(self):
        write_gguf(self.root / "a.gguf", tokens=[f"t{i}" for i in range(32)], blocks=4)
        fit = regime.vram_fit(self.root / "a.gguf", context_size=1024, parallel=2, ubatch=64, gpu_first_layer=0,
                              free_vram_bytes=10 ** 9, reserve_bytes=0, workspace_bytes=0, cuda_context_bytes=0)
        (self.root / "fit.json").write_text(json.dumps(fit))
        base = self.root / "plans.json"
        shutil.copy(DATA / "MEASURED_DESKTOP_BASELINE_PLANS_V1.json", base)
        artifact = "c" * 64
        result = regime.plans_seed(base, self.root / "plans2.json", [f"{artifact}=hot:0:{self.root / 'fit.json'}"])
        row = result["added"][0]
        self.assertEqual((row["gpu_first_layer"], row["maturity"], row["desktop_executor_id"]),
                         (0, "QUALIFIED", "physical:hot:desktop"))
        with self.assertRaises(regime.RegimeError):     # a seed at another layer than its fit
            regime.plans_seed(base, self.root / "plans3.json", [f"{artifact}=hot:2:{self.root / 'fit.json'}"])
        spec = {"schema": regime.SPEC_SCHEMA, "campaign_id_suffix": "q4gpu", "trace_dir": str(self.root / "trace"),
                "desktop_baseline_plans_path": str(self.root / "plans2.json"),
                "models": {"hot": {"model_id": "qwen3-14b-q4km-gguf", "host_artifact_path": "/m/q4.gguf",
                                   "matrix_precision": {"allowed": ["Q4_K", "Q6_K"]}, "calibration_prior": PRIOR,
                                   "phone_adapter_parameters": {"ffn_host_share_release": 0,
                                                                "ffn_host_share_populate": None,
                                                                "ffn_host_share_drop_cache": None}}}}
        (self.root / "spec.json").write_text(json.dumps(spec))
        made = regime.derive_template(PAPER / "template", self.root / "spec.json", self.root / "tpl")
        models = ModelsManifest.from_json(json.loads((self.root / "tpl/models.json").read_text()), self.root / "tpl")
        hot = next(row for row in models.models if row.model_key == "hot")
        self.assertEqual((hot.model_id, hot.phone_adapter_parameters["ffn_host_share_release"]),
                         ("qwen3-14b-q4km-gguf", 0))
        self.assertNotIn("ffn_host_share_populate", hot.phone_adapter_parameters)
        self.assertEqual(hot.calibration_prior.to_json(), PRIOR)
        campaign = json.loads((self.root / "tpl/campaign.json").read_text())
        self.assertTrue(campaign["campaign_id"].endswith("-q4gpu"))
        self.assertEqual(campaign["trace"]["trace_manifest_path"], str(self.root / "trace/TRACE_MANIFEST.json"))
        self.assertEqual(made["models"][0]["matrix_precision"], {"allowed": ["Q4_K", "Q6_K"]})

    def test_plans_seed_binds_the_catalog_placement_identity(self):
        """2026-10-06: the first all-GPU Q4 run showed that a seeded row is accepted by the catalog only with the
        operator-placement identity it derives itself, a matching cuda_graph_mode in both places, and a plans file
        that names no artifact outside the campaign."""
        from types import SimpleNamespace
        from research_dev.scheduler._internal.desktop_parent import _placement_sha256
        from research_dev.scheduler.adapters.catalog_materialization import desktop_operator_placements
        write_gguf(self.root / "a.gguf", tokens=[f"t{i}" for i in range(32)], blocks=4)
        fit = regime.vram_fit(self.root / "a.gguf", context_size=1024, parallel=2, ubatch=64, gpu_first_layer=0,
                              free_vram_bytes=10 ** 9, reserve_bytes=0, workspace_bytes=0, cuda_context_bytes=0)
        (self.root / "fit.json").write_text(json.dumps(fit))
        base = self.root / "plans.json"
        shutil.copy(DATA / "MEASURED_DESKTOP_BASELINE_PLANS_V1.json", base)
        manifest = ModelManifest.from_json(json.loads((DATA / "QWEN_MANIFEST.json").read_text()))
        artifact = manifest.artifact_sha256
        # the paper plans already hold this artifact: seed against an empty base, campaign-only output
        empty = self.root / "empty.json"
        empty.write_text(json.dumps({"schema": "s42-measured-desktop-baseline-plans-v1", "plans": []}))
        result = regime.plans_seed(empty, self.root / "seeded.json", [f"{artifact}=hot:0:{self.root / 'fit.json'}"],
                                   manifests={artifact: DATA / "QWEN_MANIFEST.json"}, cuda_graph_mode="disabled",
                                   only_new=True)
        row = result["added"][0]
        placements = desktop_operator_placements(
            manifest, 0, SimpleNamespace(cpu_device_id="desktop-cpu", gpu_device_id="desktop-cuda"))
        self.assertEqual(row["desktop_placement_sha256"], _placement_sha256(manifest, placements, "disabled"))
        self.assertEqual((row["cuda_graph_mode"], row["adapter_parameters"]["cuda_graph_mode"]), ("disabled", "disabled"))
        written = json.loads((self.root / "seeded.json").read_text())
        self.assertEqual([r["artifact_sha256"] for r in written["plans"]], [artifact])
        with self.assertRaisesRegex(regime.RegimeError, "not the seed row's artifact"):
            regime.plans_seed(empty, self.root / "seeded2.json", [f"{'c' * 64}=hot:0:{self.root / 'fit.json'}"],
                              manifests={"c" * 64: DATA / "QWEN_MANIFEST.json"})
        # without a manifest the placeholder identity is kept and labelled as such
        plain = regime.plans_seed(empty, self.root / "seeded3.json", [f"{'c' * 64}=hot:0:{self.root / 'fit.json'}"])
        self.assertIn("PLACEHOLDER", plain["added"][0]["ws17_seed"])

    def test_plans_seed_inherits_the_rig_memory_parameters_by_role(self):
        """2026-10-06 gotcha 4: the calibrated Q4 rows needed the f16 rows' memory parameters by hand; the seed now
        inherits them from the base row of the same role, the sliding-window padding only for a sliding-window model."""
        write_gguf(self.root / "a.gguf", tokens=[f"t{i}" for i in range(32)], blocks=4)
        fit = regime.vram_fit(self.root / "a.gguf", context_size=1024, parallel=2, ubatch=64, gpu_first_layer=0,
                              free_vram_bytes=10 ** 9, reserve_bytes=0, workspace_bytes=0, cuda_context_bytes=0)
        (self.root / "fit.json").write_text(json.dumps(fit))
        base = self.root / "plans.json"
        shutil.copy(DATA / "MEASURED_DESKTOP_BASELINE_PLANS_V1.json", base)
        manifests = {}
        for role, name, digit in (("cold", "GEMMA_MANIFEST.json", "a"), ("hot", "QWEN_MANIFEST.json", "b")):
            value = json.loads((DATA / name).read_text())
            value["artifact_sha256"] = "sha256:" + digit * 64
            (self.root / f"{role}.json").write_text(json.dumps(value))
            manifests[digit * 64] = self.root / f"{role}.json"
        result = regime.plans_seed(base, self.root / "seeded.json",
                                   [f"{'a' * 64}=cold:0:{self.root / 'fit.json'}",
                                    f"{'b' * 64}=hot:0:{self.root / 'fit.json'}"],
                                   manifests=manifests, cuda_graph_mode="disabled", only_new=True)
        cold, hot = result["added"]
        base_rows = {row["desktop_executor_id"]: row["adapter_parameters"]
                     for row in json.loads(base.read_text())["plans"]}
        self.assertEqual({k: v for k, v in cold["adapter_parameters"].items()
                          if k.startswith(("memory_", "kv_cache_"))},
                         {k: v for k, v in base_rows["physical:cold:desktop"].items()
                          if k.startswith(("memory_", "kv_cache_"))})
        self.assertEqual(cold["adapter_parameters"]["kv_cache_swa_padding_tokens"], 512)
        self.assertEqual(cold["adapter_parameters"]["context_size"], fit["context_size"])
        self.assertIn("inherited kv_cache_swa_padding_tokens,memory_", cold["ws17_seed"])
        self.assertEqual(hot["adapter_parameters"]["context_token_quantum"],
                         base_rows["physical:hot:desktop"]["context_token_quantum"])
        self.assertNotIn("kv_cache_swa_padding_tokens", hot["adapter_parameters"])
        # without a manifest the sliding-window padding is not inherited (no architecture evidence)
        empty_fit = regime.plans_seed(base, self.root / "seeded2.json", [f"{'c' * 64}=cold:0:{self.root / 'fit.json'}"],
                                      only_new=True)["added"][0]
        self.assertNotIn("kv_cache_swa_padding_tokens", empty_fit["adapter_parameters"])
        self.assertIn("memory_workspace_minimum_bytes:desktop-cpu", empty_fit["adapter_parameters"])

    def test_prior_receipt_weights_prefill_and_decode_by_the_trace(self):
        def bench(path, prefill_ns, decode_ns):
            path.write_text(json.dumps([{"n_prompt": 512, "n_gen": 0, "avg_ns": prefill_ns * 512},
                                        {"n_prompt": 0, "n_gen": 128, "avg_ns": decode_ns * 128}]))
        bench(self.root / "new.json", 100_000, 20_000_000)
        bench(self.root / "old.json", 200_000, 40_000_000)
        args = SimpleNamespace(new_bench=self.root / "new.json", old_bench=self.root / "old.json",
                               trace_dir=PAPER / "trace/longtail_eval_v2", tokenizer="qwen3-14b-q4_k_m",
                               assisted_new_ms_per_token=None, assisted_old_ms_per_token=None,
                               out=self.root / "r.json")
        value = regime.prior_receipt(args)
        self.assertEqual((value["desktop_service_scale_ppm"], value["assisted_service_scale_ppm"]), (500_000, 500_000))
        self.assertEqual(value["trace_rows"], 7)
        args.out = self.root / "r2.json"
        args.assisted_new_ms_per_token, args.assisted_old_ms_per_token = 300.0, 400.0
        self.assertEqual(regime.prior_receipt(args)["assisted_service_scale_ppm"], 750_000)
        args.out, args.assisted_old_ms_per_token = self.root / "r3.json", None
        with self.assertRaises(regime.RegimeError):
            regime.prior_receipt(args)
        prior = CalibrationPrior(500_000, 750_000, value["receipt_sha256"], "model")
        self.assertEqual(prior.scale(True), 0.5)

    def test_onboarding_device_spec_accepts_the_approximate_flag_only(self):
        from research_dev.scheduler.campaigns.burstgpt.tools import layer_placement

        exact = layer_placement._capability("htp:op15-phone:q4_0:9625939968:3:3208646656")
        approx = layer_placement._capability("htp:op15-phone:q4_0:9625939968:3:3208646656:approx")
        self.assertEqual((exact.allow_approximate, approx.allow_approximate), (False, True))
        self.assertEqual(replace(approx, allow_approximate=False), exact)
        with self.assertRaises(SystemExit):
            layer_placement._capability("htp:op15-phone:q4_0:9625939968:3:3208646656:maybe")


if __name__ == "__main__":
    unittest.main()
