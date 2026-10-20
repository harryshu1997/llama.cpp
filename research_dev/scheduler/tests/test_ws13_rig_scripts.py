"""WS13: desktop pieces of the batch-formation ablation and the fault-injection series.

* tools/template_variants.py derives the ``no-batch-formation`` and ``elastic-mask-out`` templates from the
  dp-online template (the frozen paper template + online device power) and the runner's own parsers execute
  exactly the intended flags;
* the composed arm (prepare -> tools/rig/derive_mpaoa.sh: measured placement + Pixel AOA) keeps those fields;
* tools/rig/launch_one_arm.sh starts the fault injector only for an attempt with a FAULTS file, passes it the
  file's condition, ends it with the arm and returns the arm's exit code; a malformed file refuses the arm.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from research_dev.scheduler.campaigns.burstgpt.tools import template_variants as variants  # noqa: E402
from research_dev.scheduler.configuration.campaign import CampaignManifest  # noqa: E402
import test_ws12_aoa_rebind_composition as ws12  # noqa: E402  (module import: its TestCase is not re-collected)

RIG = TESTS_DIR.parent / "campaigns/burstgpt/tools/rig"
FROZEN_TEMPLATE = TESTS_DIR.parent / "campaigns/burstgpt/paper_config_v1/template"
# the desktop's template-eval2-s2-dp-online = frozen template + this block (CHANGES.txt 2026-09-29)
DP_ONLINE = {
    "arrival_information": "online",
    "decode_cap": {"protect_prefill": True, "sm_max_mhz": 1200},
    "device": "desktop-cuda",
    "gpu_uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
    "idle": {"cpu_epp": None, "gpu_min_clocks_mhz": 210, "lead_ms": 500, "min_gap_s": 2},
    "load_min": True,
    "online_idle": {"fallback_idle_ms": 2000, "max_arrival_probability_ppm": 250000, "predictor": "global"},
}


def dp_online_template(root: Path) -> Path:
    template = root / "template-eval2-s2-dp-online"
    template.mkdir()
    for name in variants.TEMPLATE_FILES[1:]:
        (template / name).write_bytes((FROZEN_TEMPLATE / name).read_bytes())
    campaign = json.loads((FROZEN_TEMPLATE / "campaign.json").read_text())
    campaign["device_power"] = copy.deepcopy(DP_ONLINE)
    (template / "campaign.json").write_text(json.dumps(campaign, indent=4))
    (template / "CHANGES.txt").write_text("2026-09-29: dp-online = s2 + device_power online.\n")
    return template


class TemplateVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.base = dp_online_template(self.root)

    def test_no_batch_formation_turns_off_exactly_the_batch_formation_flags(self) -> None:
        out = self.root / "template-eval2-s2-dp-online-nobf"
        report = variants.derive(self.base, "no-batch-formation", out, today="2026-10-02")
        self.assertEqual(sorted(report["differences"]), [
            "-adaptive_decode_overrides.late_helper_adoption_minimum_tokens",
            "-dispatch_policy.max_barrier_extension_s",
            "~adaptive_decode_overrides.batch_growth_verdict_inheritance",
            "~adaptive_decode_overrides.late_helper_adoption",
            "~adaptive_decode_overrides.server_policy_coherence",
            "~dispatch_policy.continuous_join"])
        effective = report["effective"]
        self.assertEqual({name: effective["adaptive"][name] for name in variants.BATCH_FORMATION_FLAGS},
                         dict.fromkeys(variants.BATCH_FORMATION_FLAGS, False))
        self.assertEqual(effective["dispatch_policy"], {
            "continuous_join": False, "event_replanning": False, "max_barrier_extension_s": 0,
            "model_affinity": True, "residency_hysteresis_s": 20, "work_conserving_admission": True})
        self.assertTrue(effective["device_power"])
        base = variants.effective(json.loads((self.base / "campaign.json").read_text()))
        self.assertEqual(base["phone_resident_model_reprovisioning"], effective["phone_resident_model_reprovisioning"])
        self.assertTrue(all(base["adaptive"][name] for name in variants.BATCH_FORMATION_FLAGS))
        # false is each AdaptiveDecodeConfig default: the controller is the pre-feature one
        from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeConfig
        default = AdaptiveDecodeConfig()
        self.assertTrue(all(getattr(default, name) is False for name in variants.BATCH_FORMATION_FLAGS))
        for name in variants.TEMPLATE_FILES[1:]:
            self.assertEqual((out / name).read_bytes(), (self.base / name).read_bytes())
        self.assertTrue((out / "CHANGES.txt").read_text().endswith(
            "2026-10-02: template-eval2-s2-dp-online-nobf from template-eval2-s2-dp-online: no-batch-formation = "
            "base with dispatch_policy.continuous_join=false (max_barrier_extension_s dropped) and "
            "adaptive_decode_overrides server_policy_coherence / batch_growth_verdict_inheritance / "
            "late_helper_adoption = false (late_helper_adoption_minimum_tokens dropped); everything else "
            "unchanged.\n"))
        manifest = CampaignManifest.from_json(json.loads((out / "campaign.json").read_text()), out)
        self.assertIs(manifest.dispatch_policy["continuous_join"], False)
        self.assertIs(manifest.adaptive_decode_overrides["server_policy_coherence"], False)

    def test_elastic_mask_out_adds_the_g11_settings_only(self) -> None:
        out = self.root / "template-eval2-s2-dp-online-elastic"
        report = variants.derive(self.base, "elastic-mask-out", out)
        self.assertEqual(report["differences"], ["+elastic_phones"])
        self.assertEqual(report["effective"]["elastic_phones"], {
            "drop_recovery": True, "helper_loss_recovery": "mask_out", "join": True, "join_probe_interval_s": 10,
            "max_readmissions_per_device": 3, "readmission_cooldown_s": 60})
        manifest = CampaignManifest.from_json(json.loads((out / "campaign.json").read_text()), out)
        self.assertEqual(manifest.elastic_phones["helper_loss_recovery"], "mask_out")

    def test_refusals(self) -> None:
        nobf = self.root / "nobf"
        variants.derive(self.base, "no-batch-formation", nobf)
        with self.assertRaisesRegex(variants.TemplateVariantError, "does not form batches"):
            variants.derive(nobf, "no-batch-formation", self.root / "twice")
        self.assertFalse((self.root / "twice").exists())
        elastic = self.root / "elastic"
        variants.derive(self.base, "elastic-mask-out", elastic)
        with self.assertRaisesRegex(variants.TemplateVariantError, "already declares elastic_phones"):
            variants.derive(elastic, "elastic-mask-out", self.root / "again")
        with self.assertRaisesRegex(variants.TemplateVariantError, "already exists"):
            variants.derive(self.base, "elastic-mask-out", elastic)
        with self.assertRaisesRegex(variants.TemplateVariantError, "not a campaign template"):
            variants.derive(self.root, "elastic-mask-out", self.root / "none")
        # the runner refuses continuous joining without server policy coherence; so does derive's check
        campaign = json.loads((self.base / "campaign.json").read_text())
        campaign["adaptive_decode_overrides"]["server_policy_coherence"] = False
        campaign["adaptive_decode_overrides"]["batch_growth_verdict_inheritance"] = False
        with self.assertRaisesRegex(Exception, "continuous_join requires"):
            variants.effective(campaign)

    def test_command_line(self) -> None:
        out = self.root / "cli"
        completed = subprocess.run([sys.executable, "-m", "research_dev.scheduler.campaigns.burstgpt.tools."
                                    "template_variants", "derive", "--base", str(self.base), "--variant",
                                    "no-batch-formation", "--out", str(out)],
                                   cwd=TESTS_DIR.parents[2], capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        completed = subprocess.run([sys.executable, "-m", "research_dev.scheduler.campaigns.burstgpt.tools."
                                    "template_variants", "check", "--template", str(out), "--base", str(self.base)],
                                   cwd=TESTS_DIR.parents[2], capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("~dispatch_policy.continuous_join", json.loads(completed.stdout)["differences"])


class DerivedArmKeepsTemplateFieldsTests(ws12.RebindAndComposeTests):
    def test_measured_placement_and_aoa_keep_elastic_and_batch_formation_fields(self) -> None:
        evidence = self.rebind()
        base = self.base_arm()
        (base / "TRANSPORT_QUALIFICATION_IDENTITY.json").write_text("{}")
        campaign = json.loads((base / "campaign.json").read_text())
        campaign, _ = variants.no_batch_formation(campaign)
        campaign, _ = variants.elastic_mask_out(campaign)
        (base / "campaign.json").write_text(json.dumps(campaign))
        work, deploy = self.root / "placement", self.root / "deploy"
        work.mkdir()
        deploy.mkdir()
        (deploy / "source").symlink_to(TESTS_DIR.parents[2])
        (work / "DERIVE_AOA_EVIDENCE").write_text(str(evidence) + "\n")
        (work / "PLAN.json").write_text(json.dumps(self.plan(0b1)))
        (work / "DERIVE_MP_ARGS").write_text("\n".join([
            "--plan", str(work / "PLAN.json"), "--tag", "mp",
            "--primary-index", f"{ws12.QWEN}={self.root / 'OP15_QWEN_0.json'}=/phone/qwen0",
            "--helper-index", f"{ws12.QWEN}:{ws12.h.PIXEL}={self.root / 'PIXEL_FFN_SHARDS.json'}=/phone",
            "--helper-evidence", f"{ws12.QWEN}:{ws12.h.PIXEL}={evidence}"]) + "\n")
        out = self.root / "inputs-derived-mpaoa-p6f1flt"
        completed = subprocess.run(["bash", str(RIG / "derive_mpaoa.sh"), str(base), str(out)],
                                   env={**os.environ, "S43_PLACEMENT_DIR": str(work), "EVAL_DEPLOY": str(deploy)},
                                   capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        derived = json.loads((out / "campaign.json").read_text())
        self.assertEqual(derived["elastic_phones"], variants.G11_ELASTIC_PHONES)
        self.assertIs(derived["dispatch_policy"]["continuous_join"], False)
        self.assertEqual({name: derived["adaptive_decode_overrides"][name] for name in variants.BATCH_FORMATION_FLAGS},
                         dict.fromkeys(variants.BATCH_FORMATION_FLAGS, False))
        manifest = CampaignManifest.from_json(derived, out)
        self.assertEqual(manifest.elastic_phones["helper_loss_recovery"], "mask_out")
        self.assertEqual(json.loads((out / "rig.json").read_text())["helper_phones"][0]["transport"], "aoa-bridge")


for _name in [name for name in vars(ws12.RebindAndComposeTests) if name.startswith("test")]:
    setattr(DerivedArmKeepsTemplateFieldsTests, _name, None)


STUB_CHAIN = r'''import json, os, sys, time
arm = sys.argv[sys.argv.index("--arm") + 1]
os.makedirs(os.path.join(arm, "run-eval", "run"))
open(os.path.join(arm, "run-eval", "RESOLVED_CONFIGURATION.json"), "w").write("{}")
time.sleep(float(os.environ.get("STUB_SECONDS", "1.5")))
open(os.path.join(os.environ["STUB_OUT"], "chain.json"), "w").write(json.dumps(sys.argv[1:]))
sys.exit(int(os.environ.get("STUB_RC", "0")))
'''
STUB_INJECTOR = r'''import json, os, signal, sys, time
out = os.environ["STUB_OUT"]
open(os.path.join(out, "injector.json"), "w").write(json.dumps(sys.argv[1:]))
def ended(*_):
    open(os.path.join(out, "injector-terminated"), "w").write("TERM")
    sys.exit(0)
signal.signal(signal.SIGTERM, ended)
time.sleep(120)
'''


class LaunchFaultHookTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.r, self.out = root / "eval", root / "out"
        for path in (self.r / "chains", self.r / "rig" / "FAULTS", self.r / "inputs-desktop-x1",
                     self.r / "inputs-two-phone-x1", self.out):
            path.mkdir(parents=True)
        (self.r / "run_chain_eval.py").write_text(STUB_CHAIN)
        self.injector = root / "inject_stub.py"
        self.injector.write_text(STUB_INJECTOR)
        self.env = {**os.environ, "EVAL_ROOT": str(self.r), "EVAL_DEPLOY": str(root / "deploy"),
                    "EVAL_LOCK": str(root / "lock"), "S43_FAULT_INJECTOR": str(self.injector),
                    "STUB_OUT": str(self.out), "S43_FAULT_POLL_S": "0.2", "STUB_SECONDS": "3"}

    def launch(self, **env) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(RIG / "launch_one_arm.sh"), "x1", "template-x", "two-phone"],
                              env={**self.env, **env}, capture_output=True, text=True, timeout=60, check=False)

    def test_without_a_faults_file_no_injector_runs(self) -> None:
        completed = self.launch()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((self.out / "chain.json").is_file())
        self.assertFalse((self.out / "injector.json").exists())
        self.assertFalse((self.r / "chains" / "FAULT-x1.log").exists())

    def test_the_injector_gets_the_condition_and_ends_with_the_arm(self) -> None:
        (self.r / "rig" / "FAULTS" / "x1").write_text("active=500\n# authorized by the user 2026-10-01\n")
        completed = self.launch(STUB_RC="7")
        self.assertEqual(completed.returncode, 7, completed.stderr)   # the arm's exit code, not the injector's
        arm = self.r / "inputs-two-phone-x1"
        self.assertEqual(json.loads((self.out / "injector.json").read_text()), [
            "--device", "pixel10pro-phone", "--when", "active=500", "--signal", "TERM",
            "--run-dir", str(arm / "run-eval" / "run"), "--authorized", "--timeout-s", "3600"])
        self.assertTrue((self.out / "injector-terminated").is_file())
        self.assertTrue((self.r / "chains" / "FAULT-x1.log").is_file())
        self.assertIn("--arm", json.loads((self.out / "chain.json").read_text()))

    def test_a_malformed_condition_refuses_the_arm(self) -> None:
        for text in ("t=abc\n", "kill=1\n", "request=\n", "t=5; rm -rf /\n"):
            with self.subTest(text=text):
                (self.r / "rig" / "FAULTS" / "x1").write_text(text)
                completed = self.launch()
                self.assertEqual(completed.returncode, 16)
                self.assertFalse((self.out / "chain.json").exists())
                self.assertFalse((self.out / "injector.json").exists())


if __name__ == "__main__":
    unittest.main()
