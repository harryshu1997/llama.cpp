"""Derived trace inputs: the coherence override and the qualified coalesced plan are declared, not hand-edited."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from research_dev.scheduler.campaigns.burstgpt import prepare_trace_inputs_v2 as prepare

OLD = "/mnt/storage/old-deploy"
NEW = "/mnt/storage/new-deploy"


def write_source(source: Path) -> None:
    (source / "campaign.json").write_text(json.dumps({
        "campaign_id": "plain", "rig_manifest_path": str(source / "rig.json"),
        "models_manifest_path": str(source / "models.json"), "evidence_manifest_path": str(source / "evidence.json"),
        "trace": {"replay_schedule_path": "/traces/a.json", "trace_manifest_path": "/traces/a-manifest.json"},
        "adaptive_maximum_probe_attempts_per_context": 4,
    }))
    (source / "rig.json").write_text(json.dumps({
        "binaries": {"phone_boot_image": "/boot/old.img", "server": OLD + "/bin/llama-server"},
        "phone": {"boot_image_sha256": "sha256:" + "1" * 64, "serial": "S"},
    }))
    (source / "models.json").write_text(json.dumps({"models": [
        {"model_id": "qwen", "model_key": "hot", "kind": "assisted",
         "phone_adapter_parameters": {"ffn_host_share_release": 1, "usb_batch_plan": "coalesced-batch"},
         "phone_batch_plans": ["split-row", "coalesced-batch"], "qualified_phone_batch_plans": ["split-row"],
         "runtime_parameters": {"parallel": 4, "ubatch_size": 512}},
        {"model_id": "gemma", "model_key": "cold", "kind": "assisted",
         "phone_adapter_parameters": {"ffn_host_share_release": 1},
         "phone_batch_plans": ["split-row", "coalesced-batch"], "qualified_phone_batch_plans": ["split-row"],
         "runtime_parameters": {"parallel": 8, "ubatch_size": 512}},
        {"model_id": "llama", "model_key": "overlay", "kind": "overlay", "phone_batch_plans": []},
    ]}))
    (source / "evidence.json").write_text(json.dumps({
        "transport_qualification_directories": ["/receipts/old"],
        "transport_qualification_identity_path": str(source / "TRANSPORT_QUALIFICATION_IDENTITY.json"),
        "kernel_profile_path": OLD + "/profile.json",
    }))


def run(source: Path, output: Path, *extra: str) -> dict[str, dict]:
    argv = ["prepare_trace_inputs_v2.py", "--source", str(source), "--output", str(output),
            "--campaign-id", "derived", "--old-deploy", OLD, "--new-deploy", NEW, *extra]
    with patch.object(sys, "argv", argv):
        prepare.main()
    return {name: json.loads((output / (name + ".json")).read_text())
            for name in ("campaign", "rig", "models", "evidence")}


class PrepareTraceInputsTests(unittest.TestCase):
    def test_plain_derivation_changes_only_paths_ids_and_cache_policy(self):
        with TemporaryDirectory() as temporary:
            source, output = Path(temporary, "source"), Path(temporary, "output")
            source.mkdir()
            write_source(source)
            result = run(source, output)
            self.assertNotIn("adaptive_decode_overrides", result["campaign"])
            qwen = result["models"]["models"][0]
            self.assertEqual(qwen["qualified_phone_batch_plans"], ["split-row"])
            self.assertEqual(qwen["phone_adapter_parameters"]["usb_batch_plan"], "coalesced-batch")
            self.assertEqual(result["evidence"]["transport_qualification_directories"], ["/receipts/old"])
            self.assertEqual(result["evidence"]["kernel_profile_path"], NEW + "/profile.json")
            self.assertEqual(result["rig"]["phone"]["boot_image_sha256"], "sha256:" + "1" * 64)

    def test_coherence_override_and_qualified_coalesced_plan_are_declared(self):
        with TemporaryDirectory() as temporary:
            source, output = Path(temporary, "source"), Path(temporary, "output")
            source.mkdir()
            write_source(source)
            boot = Path(temporary, "candidate-boot.img")
            boot.write_bytes(b"boot")
            result = run(
                source, output,
                "--adaptive-decode-overrides-json", '{"server_policy_coherence": true}',
                "--qualify-phone-batch-plan", "hot=coalesced-batch",
                "--allow-mixed-phone-batch-plans",
                "--transport-receipts-dir", str(Path(temporary, "receipts")),
                "--phone-boot-image", str(boot), "--phone-boot-image-sha256", "sha256:" + "f" * 64,
            )
            self.assertEqual(result["campaign"]["adaptive_decode_overrides"], {"server_policy_coherence": True})
            self.assertEqual(result["campaign"]["adaptive_maximum_probe_attempts_per_context"], 4)
            qwen, gemma = result["models"]["models"][:2]
            self.assertEqual(qwen["phone_batch_plans"], ["split-row", "coalesced-batch"])
            self.assertEqual(qwen["qualified_phone_batch_plans"], ["coalesced-batch"])
            # The common override would give the unsuffixed executor the coalesced parameters too.
            self.assertNotIn("usb_batch_plan", qwen["phone_adapter_parameters"])
            self.assertEqual(gemma["qualified_phone_batch_plans"], ["split-row"])
            self.assertEqual(result["evidence"]["transport_qualification_directories"],
                             [str(Path(temporary, "receipts").resolve())])
            self.assertEqual(result["evidence"]["transport_qualification_identity_path"],
                             str(output / "TRANSPORT_QUALIFICATION_IDENTITY.json"))
            self.assertEqual(result["rig"]["phone"]["boot_image_sha256"], "sha256:" + "f" * 64)
            self.assertEqual(result["rig"]["binaries"]["phone_boot_image"], str(boot.resolve()))
            changes = (output / "CHANGES.txt").read_text()
            self.assertIn("server_policy_coherence", changes)
            self.assertIn("qualified_phone_batch_plans=['coalesced-batch']", changes)
            self.assertIn("mixed qualified phone batch plans accepted", changes)

    def test_mixed_qualified_batch_plans_across_models_are_refused_by_default(self):
        # One direct phone session serves both models and its transport contract includes the batch plan.
        with TemporaryDirectory() as temporary:
            source = Path(temporary, "source")
            source.mkdir()
            write_source(source)
            with self.assertRaises(SystemExit) as refused:
                run(source, Path(temporary, "mixed"), "--qualify-phone-batch-plan", "hot=coalesced-batch")
            self.assertIn("one phone session cannot serve both", str(refused.exception))
            self.assertFalse(Path(temporary, "mixed", "models.json").exists())
            both = run(source, Path(temporary, "both"), "--qualify-phone-batch-plan", "hot=coalesced-batch",
                       "--qualify-phone-batch-plan", "cold=coalesced-batch")
            self.assertEqual([row.get("qualified_phone_batch_plans") for row in both["models"]["models"][:2]],
                             [["coalesced-batch"], ["coalesced-batch"]])

    def test_phone_resident_model_reprovisioning_is_declared_and_exclusive_with_fixed_residency(self):
        from research_dev.scheduler.configuration.campaign import PhoneResidentModelReprovisioningConfiguration

        with TemporaryDirectory() as temporary:
            source = Path(temporary, "source")
            source.mkdir()
            write_source(source)
            self.assertNotIn("phone_resident_model_reprovisioning", run(source, Path(temporary, "off"))["campaign"])
            campaign = run(source, Path(temporary, "on"), "--phone-resident-model-reprovisioning-json", "{}")["campaign"]
            self.assertEqual(campaign["phone_resident_model_reprovisioning"], {})
            self.assertEqual(PhoneResidentModelReprovisioningConfiguration.from_json(
                campaign["phone_resident_model_reprovisioning"]), PhoneResidentModelReprovisioningConfiguration())
            self.assertIn("phone_resident_model_reprovisioning={}", Path(temporary, "on", "CHANGES.txt").read_text())
            rate = run(source, Path(temporary, "rate"), "--phone-resident-model-reprovisioning-json",
                       '{"load_bytes_per_second": 205000000}')["campaign"]
            self.assertEqual(rate["phone_resident_model_reprovisioning"], {"load_bytes_per_second": 205000000})
            with self.assertRaises(SystemExit):
                run(source, Path(temporary, "list"), "--phone-resident-model-reprovisioning-json", "[]")
            row = json.loads((source / "campaign.json").read_text())
            row["fixed_phone_residency"] = {"layouts": []}
            (source / "campaign.json").write_text(json.dumps(row))
            with self.assertRaises(SystemExit) as refused:
                run(source, Path(temporary, "fixed"), "--phone-resident-model-reprovisioning-json", "{}")
            self.assertIn("fixed phone residency", str(refused.exception))

    def test_dispatch_policy_is_declared_and_validated_by_the_campaign_manifest(self):
        from research_dev.scheduler.configuration.campaign import _dispatch_policy

        with TemporaryDirectory() as temporary:
            source = Path(temporary, "source")
            source.mkdir()
            write_source(source)
            self.assertNotIn("dispatch_policy", run(source, Path(temporary, "off"))["campaign"])
            policy = '{"work_conserving_admission": true, "model_affinity": true}'
            campaign = run(source, Path(temporary, "on"), "--dispatch-policy-json", policy)["campaign"]
            self.assertEqual(campaign["dispatch_policy"], {"work_conserving_admission": True, "model_affinity": True})
            self.assertEqual(dict(_dispatch_policy(campaign["dispatch_policy"])),
                             {"model_affinity": True, "work_conserving_admission": True})
            self.assertIn('dispatch_policy={"model_affinity": true, "work_conserving_admission": true}',
                          Path(temporary, "on", "CHANGES.txt").read_text())
            with self.assertRaises(SystemExit):
                run(source, Path(temporary, "list"), "--dispatch-policy-json", "[]")
            with self.assertRaises(SystemExit):
                run(source, Path(temporary, "empty"), "--dispatch-policy-json", "{}")

    def test_dispatch_policy_can_be_dropped_for_the_legacy_dispatcher(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary, "source")
            source.mkdir()
            write_source(source)
            policy = '{"work_conserving_admission": true, "model_affinity": true}'
            run(source, Path(temporary, "on"), "--dispatch-policy-json", policy)
            legacy = run(Path(temporary, "on"), Path(temporary, "legacy"), "--drop-dispatch-policy")
            self.assertNotIn("dispatch_policy", legacy["campaign"])
            self.assertIn("dispatch_policy removed (legacy dispatcher)",
                          Path(temporary, "legacy", "CHANGES.txt").read_text())
            with self.assertRaises(SystemExit):
                run(source, Path(temporary, "both"), "--drop-dispatch-policy", "--dispatch-policy-json", policy)

    def test_undeclared_plan_unknown_model_and_unpaired_boot_flags_are_rejected(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary, "source")
            source.mkdir()
            write_source(source)
            models = json.loads((source / "models.json").read_text())
            models["models"][1]["phone_batch_plans"] = ["split-row"]
            (source / "models.json").write_text(json.dumps(models))
            with self.assertRaises(SystemExit):
                run(source, Path(temporary, "o1"), "--qualify-phone-batch-plan", "cold=coalesced-batch")
            with self.assertRaises(SystemExit):
                run(source, Path(temporary, "o2"), "--qualify-phone-batch-plan", "warm=coalesced-batch")
            with self.assertRaises(SystemExit):
                run(source, Path(temporary, "o3"), "--qualify-phone-batch-plan", "hot=single")
            with self.assertRaises(SystemExit):
                run(source, Path(temporary, "o4"), "--phone-boot-image", str(Path(temporary, "b.img")))
            with self.assertRaises(SystemExit):
                run(source, Path(temporary, "o5"), "--adaptive-decode-overrides-json", "[]")


if __name__ == "__main__":
    unittest.main()
