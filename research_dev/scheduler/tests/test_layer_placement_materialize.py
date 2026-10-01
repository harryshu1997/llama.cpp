"""WS11 measured-placement arm materialization: plan -> arm inputs, fail-closed on indexes and evidence."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler._internal.layer_placement import solve_placement  # noqa: E402
from research_dev.scheduler._internal.layer_placement_io import problem_from_json  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt import layer_placement_rig as rig  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools.prepare_measured_placement import (  # noqa: E402
    MaterializationRefused,
    materialize,
)
from research_dev.scheduler.configuration.campaign import CampaignManifest  # noqa: E402

TEMPLATE = REPO_ROOT / "research_dev/scheduler/campaigns/burstgpt/paper_config_v1/template"
QWEN_SHA = "sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718"
GEMMA_SHA = "sha256:ed76f2183d2d1d65091986033023e6c78d27f6276c1b0c5826cc92acf73538cf"


def index(parent, records):
    return {"schema": "s42-ffn-shard-index-v1", "parent_sha256": parent, "shards": [
        {"path": f"{name}.ffn.gguf", "parent_sha256": parent, "shard_sha256": "sha256:" + f"{i:064x}",
         "layer_mask": f"{mask:016x}", "columns": 1, "n_ff": 1, "shard_bytes": size, "weight_type": "F16",
         "session_id": name} for i, (name, mask, size) in enumerate(records)]}


class MaterializeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        base = self.tmp / "base"
        base.mkdir()
        for name in ("campaign", "rig", "models", "evidence"):
            shutil.copy(TEMPLATE / (name + ".json"), base / (name + ".json"))
        rig_json = json.loads((base / "rig.json").read_text())
        rig_json["helper_phones"] = [{"device_id": rig.PIXEL, "worker_path": "/old/worker", "worker_port": 26990,
                                      "column_quantum": 4352, "max_tokens": 4, "backend": "CPU",
                                      "library_directories": ["/old"], "worker_environment": {}}]
        (base / "rig.json").write_text(json.dumps(rig_json))
        self.base = base
        profile = rig.measured_profile_v1()
        problem = problem_from_json(rig.rig_inventory_v1(profile=profile, pixel_transport="adb-tcp"), profile)
        self.plan = solve_placement(problem).to_json()
        limit = 3_208_646_656
        w = self.tmp
        (w / "qwen.json").write_text(json.dumps(index(QWEN_SHA, [("HTP0", 0x3F, limit - 4096),
                                                                 ("HTP1", 0xFC0, limit - 4096),
                                                                 ("HTP2", 0x3F000, limit - 4096)])))
        (w / "gemma27.json").write_text(json.dumps(index(GEMMA_SHA, [("HTP0", 0x1FF, 3_185_052_064),
                                                                    ("HTP1", 0x3FE00, 3_185_052_096),
                                                                    ("HTP2", 0x7FC0000, 3_185_052_096)])))
        (w / "gemma24.json").write_text(json.dumps(index(GEMMA_SHA, [("HTP0", 0xFF, 2_831_157_000),
                                                                    ("HTP1", 0xFF00, 2_831_157_000),
                                                                    ("HTP2", 0xFF0000, 2_831_157_000)])))
        pixel_mask = 0x1FC0000
        (w / "pixel.json").write_text(json.dumps(index(QWEN_SHA, [("PIXEL10PRO0", pixel_mask, 1_098_795_104)])))
        shard_sha = json.loads((w / "pixel.json").read_text())["shards"][0]["shard_sha256"]
        self.evidence = {"schema": "s42-static-helper-evidence-v1", "status": "PASS", "worker": {
            "device_id": rig.PIXEL, "layer_mask": pixel_mask, "artifact_sha256": QWEN_SHA,
            "shard_path": "/data/local/tmp/x/QWEN_PACKED_18_24.ffn.gguf",
            "expected_sha256_by_path": {"/data/local/tmp/x/QWEN_PACKED_18_24.ffn.gguf": shard_sha},
            "worker_path": "/new/worker", "phone_port": 26991, "worker_environment": {"S42_PIXEL_PACKED_WEIGHTS": "1"}}}
        (w / "evidence.json").write_text(json.dumps(self.evidence))
        self.kwargs = dict(
            primary_indexes={rig.QWEN: (w / "qwen.json", "/phone/qwen"), rig.GEMMA: (w / "gemma27.json", "/phone/g27")},
            helper_indexes={(rig.QWEN, rig.PIXEL): (w / "pixel.json", "/phone/pixel")},
            helper_evidence={(rig.QWEN, rig.PIXEL): w / "evidence.json"})

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_measured_plan_materializes_and_the_campaign_still_validates(self):
        out = self.tmp / "arm"
        receipt = materialize(self.base, self.plan, out, shadow=(self.tmp / "p.json", self.tmp / "i.json"),
                              **self.kwargs)
        self.assertEqual(receipt["planned_masks"][rig.GEMMA], {rig.OP15: "0000000007ffffff"})
        self.assertTrue(all(row["status"] in ("PASS", "REMOVED_NOT_IN_PLAN") for row in receipt["checks"]))
        models = {row["model_id"]: row for row in json.loads((out / "models.json").read_text())["models"]}
        self.assertEqual(models[rig.GEMMA]["phone_ffn_shard_index_path"], str(self.tmp / "gemma27.json"))
        self.assertEqual(models[rig.QWEN]["helper_phone_ffn_shards"][rig.PIXEL]["directory"], "/phone/pixel")
        self.assertNotIn("helper_phone_ffn_shards", models[rig.GEMMA])
        helper = json.loads((out / "rig.json").read_text())["helper_phones"][0]
        self.assertEqual((helper["worker_path"], helper["worker_port"]), ("/new/worker", 26991))
        campaign = json.loads((out / "campaign.json").read_text())
        self.assertEqual(campaign["dispatch_policy"]["measured_placement"]["mode"], "shadow")
        manifest = CampaignManifest.from_json(campaign, out)   # the opt-in key passes the campaign validator
        self.assertIn("measured_placement", manifest.dispatch_policy)

    def test_refusals_name_the_mismatch(self):
        gemma24 = dict(self.kwargs, primary_indexes={**self.kwargs["primary_indexes"],
                                                     rig.GEMMA: (self.tmp / "gemma24.json", "/phone/g24")})
        with self.assertRaisesRegex(MaterializationRefused, "primary index layers"):
            materialize(self.base, self.plan, self.tmp / "a1", **gemma24)
        bad = dict(self.evidence, worker=dict(self.evidence["worker"], layer_mask=0xFC0000))
        (self.tmp / "evidence.json").write_text(json.dumps(bad))
        with self.assertRaisesRegex(MaterializationRefused, "evidence qualified layers"):
            materialize(self.base, self.plan, self.tmp / "a2", **self.kwargs)
        (self.tmp / "evidence.json").write_text(json.dumps(dict(self.evidence, status="FAIL")))
        with self.assertRaisesRegex(MaterializationRefused, "not a PASS bundle"):
            materialize(self.base, self.plan, self.tmp / "a3", **self.kwargs)
        no_helper = dict(self.kwargs, helper_indexes={}, helper_evidence={})
        with self.assertRaisesRegex(MaterializationRefused, "no index/evidence"):
            materialize(self.base, self.plan, self.tmp / "a4", **no_helper)
        two = json.loads(json.dumps(self.plan))
        two["models"][rig.GEMMA]["owners"][rig.PIXEL] = {"layers": "26", "layer_mask": f"{1 << 26:016x}", "count": 1}
        (self.tmp / "evidence.json").write_text(json.dumps(self.evidence))
        both = dict(self.kwargs, helper_indexes={**self.kwargs["helper_indexes"],
                                                 (rig.GEMMA, rig.PIXEL): (self.tmp / "pixel.json", "/p")},
                    helper_evidence={**self.kwargs["helper_evidence"], (rig.GEMMA, rig.PIXEL): self.tmp / "evidence.json"})
        with self.assertRaisesRegex(MaterializationRefused, "two models"):
            materialize(self.base, two, self.tmp / "a5", **both)


if __name__ == "__main__":
    unittest.main()
