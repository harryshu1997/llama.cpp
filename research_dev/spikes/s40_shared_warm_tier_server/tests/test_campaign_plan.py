#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from campaign_plan import (  # noqa: E402
    CAMPAIGN_TOOL_PATHS,
    build_campaign,
    read_campaign,
)
from evidence_common import (  # noqa: E402
    EvidenceError,
    canonical_bytes,
    digest_file,
)


LOCK = {
    "campaign_plan_sha256": "8" * 64,
    "campaign_reduce_sha256": "9" * 64,
    "controller_binary_sha256": "1" * 64,
    "evidence_bundle_manifest_sha256": "2" * 64,
    "executor_bundle_manifest_sha256": "3" * 64,
    "ldd_sha256": "6" * 64,
    "native_bench_binary_sha256": "4" * 64,
    "nvidia_smi_sha256": "5" * 64,
    "physical_orchestrator_sha256": "a" * 64,
    "python_sha256": "7" * 64,
    "run_manifest_sha256": "b" * 64,
    "schema": "s40-primary-software-lock-v2",
}


def campaign(t2_repetitions: int = 1):
    return build_campaign(
        t2_repetitions,
        campaign_id="campaign-test",
        experiment_contract_sha256="a" * 64,
        software_lock=LOCK,
    )


class CampaignPlanTests(unittest.TestCase):
    def test_primary_is_prospectively_alternating(self):
        rows = campaign()["primary"][:12]
        self.assertEqual([row["order"] for row in rows], list(range(12)))
        for left, right in zip(rows, rows[1:]):
            self.assertNotEqual(left["mode"], right["mode"])

    def test_primary_counts_both_c1_cache_regimes(self):
        rows = campaign()["primary"][:12]
        counts = Counter(
            (row["mode"], row["cache_regime"]) for row in rows)
        self.assertEqual(
            counts[("C1_GPU_ONLY_OPTIMIZED", "WARM_HOST_CACHE")], 3)
        self.assertEqual(
            counts[("C1_GPU_ONLY_OPTIMIZED", "COLD_NVME")], 3)
        self.assertEqual(
            counts[("C2_GPU_PLUS_CPU_WARM_EXECUTOR", "WARM_HOST_CACHE")], 3)
        self.assertEqual(
            counts[("T1_PHONE_WARM_TIER", "WARM_HOST_CACHE")], 3)

    def test_t2_count_is_frozen_before_campaign(self):
        self.assertEqual(
            sum(row["mode"] == "T2_PHONE_NO_PROMOTION"
                for row in campaign(1)["primary"]),
            1,
        )
        self.assertEqual(
            sum(row["mode"] == "T2_PHONE_NO_PROMOTION"
                for row in campaign(3)["primary"]),
            3,
        )

    def test_invalid_t2_count_is_rejected(self):
        with self.assertRaisesRegex(EvidenceError, "1 or 3"):
            campaign(2)

    def test_run_ids_and_software_are_frozen(self):
        value = campaign()
        self.assertEqual(value["schema"], "s40-physical-campaign-v3")
        self.assertEqual(value["experiment_contract_sha256"], "a" * 64)
        self.assertEqual(value["software_lock"], LOCK)
        self.assertEqual(
            [row["run_id"] for row in value["primary"]],
            [
                f"campaign-test-{index:03d}"
                for index in range(len(value["primary"]))
            ],
        )

    def test_invalid_campaign_identity_and_lock_are_rejected(self):
        with self.assertRaisesRegex(EvidenceError, "lowercase ASCII"):
            build_campaign(
                1,
                campaign_id="UPPER CASE",
                experiment_contract_sha256="a" * 64,
                software_lock=LOCK,
            )
        broken = dict(LOCK)
        broken["controller_binary_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(EvidenceError, "invalid SHA-256"):
            build_campaign(
                1,
                campaign_id="campaign-test",
                experiment_contract_sha256="a" * 64,
                software_lock=broken,
            )

    def test_campaign_tool_substitution_is_rejected(self):
        lock = dict(LOCK)
        lock.update({
            f"{name}_sha256": digest_file(path)
            for name, path in CAMPAIGN_TOOL_PATHS.items()
        })
        value = build_campaign(
            1,
            campaign_id="campaign-test",
            experiment_contract_sha256="a" * 64,
            software_lock=lock,
        )
        with tempfile.TemporaryDirectory(
                prefix="s40_campaign_plan_") as directory:
            path = Path(directory) / "campaign.json"
            path.write_bytes(canonical_bytes(value))
            read_campaign(path)
            value["software_lock"][
                "physical_orchestrator_sha256"] = "0" * 64
            path.write_bytes(canonical_bytes(value))
            with self.assertRaisesRegex(
                    EvidenceError, "changed physical_orchestrator"):
                read_campaign(path)


if __name__ == "__main__":
    unittest.main()
