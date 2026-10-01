"""Per-shard arm inputs: only the trace (and campaign id) of a materialized arm directory change."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from research_dev.scheduler.campaigns.burstgpt.quality.derive_inputs import DeriveInputsError, derive
from research_dev.scheduler.configuration.campaign import CampaignManifest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import quality_fixtures as fx  # noqa: E402

FROZEN_TEMPLATE = (TESTS_DIR.parent / "campaigns/burstgpt/paper_config_v1/template/campaign.json")


class QualityDeriveInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / "inputs-two-phone-s2a"
        self.source.mkdir()
        campaign = json.loads(FROZEN_TEMPLATE.read_text())
        for name in ("rig", "models", "evidence"):
            campaign[name + "_manifest_path"] = str(self.source / (name + ".json"))
            (self.source / (name + ".json")).write_text(json.dumps({"kind": name, "outside": "/mnt/storage/x"}))
        (self.source / "evidence.json").write_text(json.dumps({
            "transport_qualification_identity_path": str(self.source / "TRANSPORT_QUALIFICATION_IDENTITY.json"),
            "stores": [str(self.source / "sub" / "a.json"), "/home/zhihao/other.json"]}))
        (self.source / "TRANSPORT_QUALIFICATION_IDENTITY.json").write_text("{}")
        (self.source / "campaign.json").write_text(json.dumps(campaign))
        self.suite = self.root / "suite"
        fx.build_small_suite(self.suite, shards=1)
        self.shard = self.suite / "shard-00"

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_the_trace_and_the_id_change(self):
        target = self.root / "q" / "two-phone-s00"
        derive(self.source, self.shard, target)
        before = json.loads((self.source / "campaign.json").read_text())
        after = json.loads((target / "campaign.json").read_text())
        changed = {name for name in set(before) | set(after) if before.get(name) != after.get(name)}
        self.assertEqual(changed, {"trace", "campaign_id", "rig_manifest_path", "models_manifest_path",
                                   "evidence_manifest_path"})
        self.assertEqual(after["campaign_id"], before["campaign_id"] + "-qtest_s00")
        self.assertEqual(after["models_manifest_path"], str(target / "models.json"))
        self.assertEqual(after["trace"]["replay_schedule_path"], str(self.shard / "qtest_s00.json"))
        evidence = json.loads((target / "evidence.json").read_text())
        self.assertEqual(evidence["transport_qualification_identity_path"],
                         str(target / "TRANSPORT_QUALIFICATION_IDENTITY.json"))
        self.assertEqual(evidence["stores"], [str(target / "sub" / "a.json"), "/home/zhihao/other.json"])
        manifest = CampaignManifest.from_json(after, target)
        self.assertEqual(manifest.trace.trace_manifest_path, self.shard / "TRACE_MANIFEST.json")
        self.assertEqual(manifest.dispatch_policy, CampaignManifest.from_json(before, self.source).dispatch_policy)

    def test_overrides_merge_but_never_replace_the_trace(self):
        target = self.root / "forced-s00"
        derive(self.source, self.shard, target, {"minimum_energy_saving_ppm": 0})
        self.assertEqual(json.loads((target / "campaign.json").read_text())["minimum_energy_saving_ppm"], 0)
        with self.assertRaises(DeriveInputsError):
            derive(self.source, self.shard, self.root / "other", {"trace": {}})
        with self.assertRaises(DeriveInputsError):
            derive(self.source, self.shard, target)
        with self.assertRaises(DeriveInputsError):
            derive(self.source, self.suite, self.root / "not-a-shard")


if __name__ == "__main__":
    unittest.main()
