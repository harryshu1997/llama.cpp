"""WS12: a qualified AOA bundle re-qualified for a new Pixel layer mask, then composed into an arm.

``prepare_aoa_evidence rebind-shard`` swaps the shard and the two receipts that depend on it;
``prepare_measured_placement`` gives the Pixel the planned layers; ``prepare_aoa_evidence derive-arm`` switches
its link to the AOA bridge. The composed arm must load with the scheduler's own loaders and the co-helper
declaration must match its evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from research_dev.scheduler.adapters.contracts import PhysicalAdapterError  # noqa: E402
from research_dev.scheduler.adapters.phone_aoa_session import AoaBridgeConfiguration  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt import catalog as campaign_catalog  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.helper_phone_evidence import (  # noqa: E402
    digest, load_helper_evidence, validate_helper_declaration,
)
from research_dev.scheduler.campaigns.burstgpt.layer_placement_rig import QWEN  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import prepare_aoa_evidence as aoa_tool  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools.prepare_measured_placement import (  # noqa: E402
    MaterializationRefused, materialize,
)
from research_dev.scheduler.configuration.campaign import CampaignManifest  # noqa: E402
from research_dev.scheduler.configuration.models import ModelsManifest  # noqa: E402
from research_dev.scheduler.configuration.rig import RigManifest  # noqa: E402

import two_phone_harness as h  # noqa: E402
import test_aoa_bridge_wiring as wiring  # noqa: E402
from test_two_phone_helpers import two_phone_rig_json  # noqa: E402

TEMPLATE = Path(__file__).resolve().parents[1] / "campaigns/burstgpt/paper_config_v1/template"
NEW_LAYERS = (1, 2, 3, 4, 5)
NEW_MASK = sum(1 << layer for layer in NEW_LAYERS)
NEW_PATH = "/phone/shard_1_5.ffn.gguf"
NEW_SHA = "sha256:" + "e" * 64


class RebindAndComposeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.evidence = aoa_tool.build_bundle(wiring.PrepareAoaEvidenceTests().fixture(self.directory.name))
        raw = json.loads(self.evidence.read_text())
        self.worker = raw["worker"]
        self.aoa = AoaBridgeConfiguration.from_json(raw["aoa_bridge"])
        (self.root / "PIXEL_FFN_SHARDS.json").write_text(json.dumps({
            "schema": "s42-ffn-shard-index-v1", "parent_sha256": h.EVIDENCE, "shards": [{
                "path": Path(NEW_PATH).name, "parent_sha256": h.EVIDENCE, "shard_sha256": NEW_SHA,
                "layer_mask": f"{NEW_MASK:016x}", "columns": 128, "n_ff": 128, "shard_bytes": 4096,
                "weight_type": "MIXED_Q4_K_Q6_K", "session_id": "PIXEL0"}]}))
        pins = {**{path: value for path, value in self.worker["expected_sha256_by_path"].items()
                   if path != self.worker["shard_path"]}, NEW_PATH: NEW_SHA}
        server = self.root / "server-identity-aoa-new"
        server.mkdir()
        (server / "RESULT.json").write_text(json.dumps({
            "status": "PASS", "token_identity": [True] * 4, "transport": "aoa-bridge", "server_exit": 0,
            "worker_exit": 0, "phone_calls": 40, "boot_unchanged": True, "output_tokens_each": 64,
            "call_columns": {"8704": 20, "17408": 20}}))
        (server / "CONFIG.json").write_text(json.dumps({
            "phone_worker": self.worker["worker_path"], "phone_model": NEW_PATH,
            "phone_library_dir": self.worker["library_directories"][0], "phone_environment": {},
            "phone": {"layers": list(NEW_LAYERS), "artifact_sha256": h.EVIDENCE, "columns": 128}}))
        (server / "IDENTITY.json").write_text(json.dumps({
            "phone_hashes": "".join(f"{pin[7:]}  {path}\n" for path, pin in pins.items()),
            "server_sha256": digest(self.root / "server")[7:],
            "aoa_bridge": {"options_sha256": self.aoa.options_sha256, "configuration": self.aoa.to_json()}}))
        suite = self.root / "numerical-new"
        suite.mkdir()
        (suite / "SUITE_RESULT.json").write_text(json.dumps({"status": "PASS", "arms": {aoa_tool.NUMERICAL_ARM: {
            "numerical_status": "PASS", "max_relative_l2": 0.0006,
            "arm": {"model_path": NEW_PATH, "model_sha256": NEW_SHA[7:]}}}}))
        (suite / "SUITE_CALLS.json").write_text(json.dumps({aoa_tool.NUMERICAL_ARM: [
            {"layer": layer, "tokens": tokens} for layer in NEW_LAYERS for tokens in (1, 2, 4)]}))
        (suite / "EXPECTED_PHONE_HASHES.json").write_text(json.dumps({path: pin[7:] for path, pin in pins.items()}))
        self.args = dict(evidence=str(self.evidence), shard_index=str(self.root / "PIXEL_FFN_SHARDS.json"),
                         phone_shard_path=NEW_PATH, server_identity=str(server), numerical_dir=str(suite),
                         carry_numerical=False, output_dir=str(self.root / "rebound"))

    def rebind(self, **change) -> Path:
        return aoa_tool.rebind_shard(argparse.Namespace(**{**self.args, **change}))

    def test_rebind_repins_a_new_worker_binary_only_when_the_identity_run_verified_it(self) -> None:
        """2026-10-02 rig: the fused kernel ships in a new worker binary; rebind-shard refused the bundle because the
        identity run used it. --worker re-pins exactly the file the identity run used and hash-verified."""
        new_worker, new_sha = "/phone/fused/llama-ffn-split-worker", "sha256:" + "d" * 64
        server = Path(self.args["server_identity"])
        config = json.loads((server / "CONFIG.json").read_text())
        (server / "CONFIG.json").write_text(json.dumps({**config, "phone_worker": new_worker}))
        identity = json.loads((server / "IDENTITY.json").read_text())
        listing = identity["phone_hashes"] + f"{new_sha[7:]}  {new_worker}\n"
        (server / "IDENTITY.json").write_text(json.dumps({**identity, "phone_hashes": listing}))
        with self.assertRaisesRegex(PhysicalAdapterError, "another worker configuration"):
            self.rebind(output_dir=str(self.root / "no-worker"))
        with self.assertRaisesRegex(PhysicalAdapterError, "did not verify this worker binary"):
            self.rebind(worker=f"{new_worker}=sha256:{'e' * 64}", output_dir=str(self.root / "wrong-sha"))
        self.assertFalse((self.root / "wrong-sha").exists())
        # an FFN-only shard on a new binary needs fresh numerics: the old suite does not cover it, carrying is refused
        with self.assertRaisesRegex(PhysicalAdapterError, "neither in the suite nor bridged"):
            self.rebind(worker=f"{new_worker}={new_sha}", output_dir=str(self.root / "new-worker"))
        with self.assertRaisesRegex(PhysicalAdapterError, "cannot carry the old numerical receipt"):
            self.rebind(worker=f"{new_worker}={new_sha}", numerical_dir=None, carry_numerical=True,
                        output_dir=str(self.root / "new-worker-carry"))
        self.assertFalse((self.root / "new-worker-carry").exists())

    def test_rebind_swaps_only_the_shard_and_its_receipts(self) -> None:
        old = load_helper_evidence(self.evidence)
        new = load_helper_evidence(self.rebind())
        self.assertEqual(new.worker.layer_mask, NEW_MASK)
        self.assertEqual((new.worker.shard_path, new.worker.expected_sha256_by_path[NEW_PATH]), (NEW_PATH, NEW_SHA))
        self.assertNotIn(self.worker["shard_path"], new.worker.expected_sha256_by_path)
        self.assertEqual(new.identity.software_identity["phone_shard_sha256"], NEW_SHA)
        self.assertEqual(new.identity.software_identity["phone_worker_sha256"],
                         old.identity.software_identity["phone_worker_sha256"])
        self.assertEqual(new.aoa_bridge, old.aoa_bridge)
        changed = {kind for kind in old.identity.receipts if old.identity.receipts[kind] != new.identity.receipts[kind]}
        self.assertEqual(changed, {"server-token-identity", "numerical-rows-1-2-4"})
        numerical = json.loads(Path(json.loads(
            (self.root / "rebound" / "PIXEL_EVIDENCE_AOA.json").read_text())["receipt_paths"]["numerical-rows-1-2-4"]
        ).read_text())
        self.assertEqual(numerical["layers"], list(NEW_LAYERS))

    def test_rebind_refusals(self) -> None:
        suite = Path(self.args["numerical_dir"])
        server = Path(self.args["server_identity"])
        cases = (
            ("layers", lambda: (server / "CONFIG.json").write_text(json.dumps({
                **json.loads((server / "CONFIG.json").read_text()),
                "phone": {"layers": [4, 5], "artifact_sha256": h.EVIDENCE, "columns": 128}})),
             "served other layers"),
            ("coverage", lambda: (suite / "SUITE_CALLS.json").write_text(json.dumps({aoa_tool.NUMERICAL_ARM: [
                {"layer": layer, "tokens": tokens} for layer in (4, 5) for tokens in (1, 2, 4)]})),
             "exactly the new layers"),
            ("shard", lambda: (suite / "SUITE_RESULT.json").write_text(json.dumps({"status": "PASS", "arms": {
                aoa_tool.NUMERICAL_ARM: {"numerical_status": "PASS", "max_relative_l2": 0.0006,
                                         "arm": {"model_path": "/phone/other.gguf", "model_sha256": NEW_SHA[7:]}}}})),
             "another shard"),
            ("tokens", lambda: (server / "RESULT.json").write_text(json.dumps({
                **json.loads((server / "RESULT.json").read_text()), "token_identity": [True, False, True, True]})),
             "server identity over the bridge failed"),
        )
        for name, mutate, message in cases:
            with self.subTest(case=name):
                backup = {path: path.read_text() for path in (*suite.iterdir(), *server.iterdir())}
                mutate()
                with self.assertRaisesRegex(PhysicalAdapterError, message):
                    self.rebind()
                self.assertFalse((self.root / "rebound").exists())
                for path, text in backup.items():
                    path.write_text(text)

    def test_identity_config_changes_only_the_shard_and_layers(self) -> None:
        base = {"phone_model": self.worker["shard_path"], "phone_worker": self.worker["worker_path"],
                "expected_phone_sha256": {path: value[7:] for path, value in self.worker["expected_sha256_by_path"].items()},
                "phone": {"layers": [4, 5], "artifact_sha256": h.EVIDENCE, "columns": 128, "n_embd": 32},
                "server": "/srv/llama-server", "output_tokens": 64}
        (self.root / "BASE_CONFIG.json").write_text(json.dumps(base))
        out = aoa_tool.identity_config(argparse.Namespace(
            base=str(self.root / "BASE_CONFIG.json"), shard_index=self.args["shard_index"], phone_shard_path=NEW_PATH,
            output=str(self.root / "NEW_CONFIG.json")))
        config = json.loads(out.read_text())
        self.assertEqual((config["phone_model"], config["phone"]["layers"]), (NEW_PATH, list(NEW_LAYERS)))
        self.assertNotIn(self.worker["shard_path"], config["expected_phone_sha256"])
        self.assertEqual(config["expected_phone_sha256"][NEW_PATH], NEW_SHA[7:])
        for key in ("phone_worker", "server", "output_tokens"):
            self.assertEqual(config[key], base[key])
        other = {**base, "phone": {**base["phone"], "artifact_sha256": "sha256:" + "0" * 64}}
        (self.root / "OTHER_CONFIG.json").write_text(json.dumps(other))
        with self.assertRaisesRegex(PhysicalAdapterError, "not a shard of the qualified artifact"):
            aoa_tool.identity_config(argparse.Namespace(
                base=str(self.root / "OTHER_CONFIG.json"), shard_index=self.args["shard_index"],
                phone_shard_path=NEW_PATH, output=str(self.root / "NEW2.json")))
        self.assertFalse((self.root / "NEW2.json").exists())

    def test_identity_config_moves_the_server_layer_mask_with_the_shard(self) -> None:
        """2026-10-02 rig: a config keeping the old S41_SERVER_FFN_LAYER_MASK failed with 'FFN control exceeds
        resident slice' at the first helper control; the server mask must follow the new shard's layers."""
        base = {"phone_model": self.worker["shard_path"], "phone_worker": self.worker["worker_path"],
                "expected_phone_sha256": {path: value[7:] for path, value in self.worker["expected_sha256_by_path"].items()},
                "phone": {"layers": [4, 5], "artifact_sha256": h.EVIDENCE, "columns": 128, "n_embd": 32},
                "server": "/srv/llama-server", "output_tokens": 64,
                "server_environment": {"S41_SERVER_FFN_LAYER_MASK": str((1 << 4) | (1 << 5)),
                                       "S41_SERVER_FFN_COLUMNS": "128"}}
        (self.root / "BASE_MASK.json").write_text(json.dumps(base))
        out = aoa_tool.identity_config(argparse.Namespace(
            base=str(self.root / "BASE_MASK.json"), shard_index=self.args["shard_index"], phone_shard_path=NEW_PATH,
            output=str(self.root / "NEW_MASK.json")))
        environment = json.loads(out.read_text())["server_environment"]
        self.assertEqual(environment["S41_SERVER_FFN_LAYER_MASK"], str(NEW_MASK))
        self.assertEqual(environment["S41_SERVER_FFN_COLUMNS"], "128")
        stale = {**base, "server_environment": {**base["server_environment"], "S41_SERVER_FFN_LAYER_MASK": "7"}}
        (self.root / "STALE.json").write_text(json.dumps(stale))
        with self.assertRaisesRegex(PhysicalAdapterError, "base server layer mask differs"):
            aoa_tool.identity_config(argparse.Namespace(
                base=str(self.root / "STALE.json"), shard_index=self.args["shard_index"],
                phone_shard_path=NEW_PATH, output=str(self.root / "STALE_OUT.json")))
        self.assertFalse((self.root / "STALE_OUT.json").exists())

    def test_probe_config_runs_every_new_layer_on_timing_inputs(self) -> None:
        captures = self.root / "inputs"
        captures.mkdir()
        for row in range(4):
            (captures / f"input-layer4-row{row}.f16").write_bytes(bytes([row]) * 64)
        base = {"phone": {"artifact_sha256": h.EVIDENCE, "columns": 128, "layers": [4, 5], "shard_path": "/phone/old.gguf",
                          "common_sha256": {"/phone/old.gguf": "sha256:" + "1" * 64, "/phone/lib.so": "sha256:" + "2" * 64}},
                "input_dir": str(captures), "cadence": {"calls_per_token": 2, "gap_call_ms": 6.0},
                "segments": [], "arms": [{"name": "aoa"}, {"name": "aoa-qos"}], "calibration_arm": "aoa-qos",
                "repeats": 2}
        (self.root / "AB.json").write_text(json.dumps(base))
        out = aoa_tool.probe_config(argparse.Namespace(
            base=str(self.root / "AB.json"), shard_index=self.args["shard_index"], phone_shard_path=NEW_PATH,
            input_dir=str(self.root / "probe-inputs"), tokens="1=1200,2=300", output=str(self.root / "PROBE.json")))
        config = json.loads(out.read_text())
        self.assertEqual((config["phone"]["layers"], config["cadence"]["calls_per_token"]), (list(NEW_LAYERS), 5))
        self.assertEqual(config["phone"]["common_sha256"], {"/phone/lib.so": "sha256:" + "2" * 64, NEW_PATH: NEW_SHA})
        self.assertEqual([arm["name"] for arm in config["arms"]], ["aoa-qos"])
        self.assertEqual([(row["rows"], row["tokens"]) for row in config["segments"]], [(1, 1200), (2, 300)])
        self.assertEqual((self.root / "probe-inputs" / "input-layer1-row1.f16").read_bytes(), bytes([1]) * 64)
        self.assertEqual(config["timing_inputs"]["copied_from_layer"], 4)

    def test_carry_numerical_is_explicit_and_annotated(self) -> None:
        path = self.rebind(numerical_dir=None, carry_numerical=True)
        raw = json.loads(path.read_text())
        numerical = json.loads(Path(raw["receipt_paths"]["numerical-rows-1-2-4"]).read_text())
        self.assertEqual(numerical["carried_over"]["layer_mask"], NEW_MASK)
        self.assertEqual(raw["rebound_from"]["numerical"], "carried-over")
        load_helper_evidence(path)

    def base_arm(self) -> Path:
        base = self.root / "base"
        base.mkdir()
        rig = json.loads(json.dumps(two_phone_rig_json()).replace("pixel10pro-phone", h.PIXEL))
        rig["helper_phones"][0]["forward_port"] = 26991
        campaign = json.loads((TEMPLATE / "campaign.json").read_text())
        for name in ("rig", "models", "evidence"):
            campaign[name + "_manifest_path"] = str(base / (name + ".json"))
        (base / "campaign.json").write_text(json.dumps(campaign))
        (base / "rig.json").write_text(json.dumps(rig))
        (base / "models.json").write_text((TEMPLATE / "models.json").read_text())
        (base / "evidence.json").write_text((TEMPLATE / "evidence.json").read_text())
        (self.root / "OP15_QWEN_0.json").write_text(json.dumps({
            "schema": "s42-ffn-shard-index-v1", "parent_sha256": h.EVIDENCE, "shards": [{
                "path": "HTP0.ffn.gguf", "parent_sha256": h.EVIDENCE, "shard_sha256": "sha256:" + "f" * 64,
                "layer_mask": f"{1:016x}", "columns": 128, "n_ff": 128, "shard_bytes": 4096, "weight_type": "F16",
                "session_id": "HTP0"}]}))
        return base

    def plan(self, op15_mask: int) -> dict:
        owners = {h.PIXEL: {"layer_mask": f"{NEW_MASK:016x}", "count": len(NEW_LAYERS)}}
        owners["op15-phone"] = {"layer_mask": f"{op15_mask:016x}", "count": bin(op15_mask).count("1")}
        return {"schema": "research-scheduler-layer-placement-v1", "models": {QWEN: {"owners": owners}}}

    def test_materialize_then_derive_arm_loads_and_matches_its_evidence(self) -> None:
        evidence = self.rebind()
        base = self.base_arm()
        kwargs = dict(primary_indexes={QWEN: (self.root / "OP15_QWEN_0.json", "/phone/qwen0")},
                      helper_indexes={(QWEN, h.PIXEL): (self.root / "PIXEL_FFN_SHARDS.json", "/phone")},
                      helper_evidence={(QWEN, h.PIXEL): evidence})
        with self.assertRaisesRegex(MaterializationRefused, "primary phone op15-phone none"):
            materialize(base, self.plan(0), self.root / "unanchored", **kwargs)
        arm = self.root / "arm-mp"
        materialize(base, self.plan(0b1), arm, tag="mp", **kwargs)
        target = aoa_tool.derive_arm(argparse.Namespace(inputs=str(arm), evidence=str(evidence),
                                                        output=str(self.root / "arm-mp-aoa")))
        campaign = json.loads((target / "campaign.json").read_text())
        self.assertTrue(campaign["campaign_id"].endswith("-mp-aoa"))
        CampaignManifest.from_json(campaign, target)
        rig = RigManifest.from_json(json.loads((target / "rig.json").read_text()), target)
        models = ModelsManifest.from_json(json.loads((target / "models.json").read_text()), target)
        paths = json.loads((target / "evidence.json").read_text())["helper_phone_evidence_paths"]
        self.assertEqual(paths, {h.PIXEL: str(evidence.resolve())})
        loaded = load_helper_evidence(paths[h.PIXEL])
        row = next(value for value in rig.helper_phones if value.device_id == h.PIXEL)
        self.assertEqual((row.transport, row.worker_path), ("aoa-bridge", self.worker["worker_path"]))
        qwen = next(value for value in models.models if value.model_id == QWEN)
        manifest = SimpleNamespace(model_id=QWEN, artifact_sha256=h.EVIDENCE, feed_forward_length=128,
                                   embedding_length=32)
        declaration = campaign_catalog.helper_phone_co_helpers(rig, qwen, manifest)
        self.assertEqual(declaration.layer_mask, NEW_MASK)
        self.assertEqual(declaration.helpers[0].transport_parameters["ffn_link_transport"], "aoa-bridge")
        validate_helper_declaration(declaration, manifest, loaded, row)

    def test_rig_derive_script_composes_the_arm_and_cleans_up_a_refusal(self) -> None:
        """tools/rig/derive_mpaoa.sh (launch_one_arm.sh KIND=derived-mpaoa) runs the same two steps from files."""
        import os
        import subprocess
        evidence = self.rebind()
        base = self.base_arm()
        (base / "TRANSPORT_QUALIFICATION_IDENTITY.json").write_text("{}")
        work, deploy = self.root / "placement", self.root / "deploy"
        work.mkdir()
        deploy.mkdir()
        (deploy / "source").symlink_to(TESTS_DIR.parents[2])
        (work / "DERIVE_AOA_EVIDENCE").write_text(str(evidence) + "\n")
        script = TESTS_DIR.parent / "campaigns/burstgpt/tools/rig/derive_mpaoa.sh"
        env = {**os.environ, "S43_PLACEMENT_DIR": str(work), "EVAL_DEPLOY": str(deploy)}

        def run(op15_mask: int, out: Path) -> subprocess.CompletedProcess:
            (work / "PLAN.json").write_text(json.dumps(self.plan(op15_mask)))
            (work / "DERIVE_MP_ARGS").write_text("\n".join([
                "--plan", str(work / "PLAN.json"), "--tag", "mp",
                "--primary-index", f"{QWEN}={self.root / 'OP15_QWEN_0.json'}=/phone/qwen0",
                "--helper-index", f"{QWEN}:{h.PIXEL}={self.root / 'PIXEL_FFN_SHARDS.json'}=/phone",
                "--helper-evidence", f"{QWEN}:{h.PIXEL}={evidence}"]) + "\n")
            return subprocess.run(["bash", str(script), str(base), str(out)], env=env, capture_output=True,
                                  text=True, timeout=120)

        refused = self.root / "inputs-derived-mpaoa-x0"
        result = run(0, refused)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(refused.exists() or Path(str(refused) + ".mp").exists(), result.stderr)
        out = self.root / "inputs-derived-mpaoa-x1"
        result = run(0b1, out)
        self.assertEqual(result.returncode, 0, result.stderr)
        campaign = json.loads((out / "campaign.json").read_text())
        self.assertTrue(campaign["campaign_id"].endswith("-mp-aoa"))
        CampaignManifest.from_json(campaign, out)
        rig = RigManifest.from_json(json.loads((out / "rig.json").read_text()), out)
        row = next(value for value in rig.helper_phones if value.device_id == h.PIXEL)
        self.assertEqual(row.transport, "aoa-bridge")
        self.assertTrue((out / "TRANSPORT_QUALIFICATION_IDENTITY.json").is_file())
        self.assertEqual(run(0b1, out).returncode, 3)    # never overwrites an arm


if __name__ == "__main__":
    unittest.main()
