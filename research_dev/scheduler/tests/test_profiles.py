#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research_dev.scheduler import (
    LifecycleReceipt,
    MatmulScheduleError,
    MatmulSystemProfile,
    ProfileCatalogError,
    ProfileBundle,
    ResourceProfile,
    UnifiedScheduleError,
    UnifiedScheduler,
    load_lifecycle_profile_set,
    load_profile_bundle,
    materialize_matmul_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
S42_ROOT = REPO_ROOT / "research_dev/spikes/s42_general_energy_scheduler_v1"
BGE_ROOT = S42_ROOT / "small_model_phone_v1/results/4060ti_op15_20260808"


class ProfileCatalogTests(unittest.TestCase):
    def test_lifecycle_catalog_owns_state_selection(self) -> None:
        raw, profile_set = load_lifecycle_profile_set(
            "bge-lifecycle-test",
            "cuda0",
            {
                "cuda_epoch_open": (
                    BGE_ROOT / "SCHEDULER_PROFILE_CUDA_EPOCH_OPEN.json"
                ),
                "cuda_epoch_reused": (
                    BGE_ROOT / "SCHEDULER_PROFILE_CUDA_EPOCH_REUSED.json"
                ),
            },
            "cuda_epoch_open",
            frozenset({"cuda_epoch_reused"}),
        )
        self.assertEqual(set(raw), {"cuda_epoch_open", "cuda_epoch_reused"})
        self.assertEqual(profile_set.select_state(None), "cuda_epoch_open")
        self.assertEqual(
            profile_set.select_state(LifecycleReceipt("unrecognized")),
            "cuda_epoch_open",
        )
        self.assertEqual(
            profile_set.select_state(LifecycleReceipt(
                "cuda_epoch_reused", "tail-charge-1"
            )),
            "cuda_epoch_reused",
        )
        with self.assertRaises(UnifiedScheduleError):
            profile_set.select_state(LifecycleReceipt("cuda_epoch_reused"))

    def test_profile_hash_mismatch_is_rejected(self) -> None:
        source = BGE_ROOT / "SCHEDULER_PROFILE_CUDA_EPOCH_OPEN.json"
        value = json.loads(source.read_text(encoding="ascii"))
        value["profile_id"] += "-changed"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(value) + "\n", encoding="ascii")
            with self.assertRaisesRegex(ProfileCatalogError, "hash mismatch"):
                load_profile_bundle(path)


class ProfileMaterializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = json.loads(
            (S42_ROOT / "MEASURED_4060TI_OP15_KERNEL_PROFILE_V1.json").read_text(
                encoding="ascii"
            )
        )

    def test_all_measured_compute_backends_reach_the_scheduler(self) -> None:
        value = materialize_matmul_profile(self.source)
        profile = MatmulSystemProfile.from_json(value)
        qualification = value["qualification"]
        self.assertEqual(
            qualification["included_backends"],
            ["adreno", "cpu", "cuda", "htp"],
        )
        phone_rows = {
            row.profile_id: row for row in profile.kernels
            if row.device_id == "phone"
        }
        self.assertTrue(any(name.startswith("htp-") for name in phone_rows))
        self.assertTrue(any(name.startswith("adreno-") for name in phone_rows))
        self.assertTrue(all(
            row.maximum_n == 512
            for name, row in phone_rows.items()
            if name.startswith("adreno-")
        ))
        self.assertTrue(all(
            row.maximum_n == 11_136
            for name, row in phone_rows.items()
            if name.startswith("htp-")
        ))

    def test_materializer_can_bind_the_shared_runtime_calendar(self) -> None:
        shared = {
            "cpu": ResourceProfile("cpu-cold", "cpu", 2, True, "i9-12900K"),
            "gpu": ResourceProfile("cuda0", "gpu", 8, True, "rtx-4060-ti"),
            "phone": ResourceProfile(
                "op15-phone-compute", "phone", 1, True, "op15"
            ),
            "pcie": ResourceProfile(
                "pcie-root-0", "transport", 1, True, "pcie-gen1-x8"
            ),
            "usb": ResourceProfile(
                "op15-usb", "transport", 1, True, "usb-5gbps"
            ),
        }
        value = materialize_matmul_profile(
            self.source,
            resource_profiles=shared,
        )
        profile = MatmulSystemProfile.from_json(value)
        self.assertEqual(profile.devices["cpu"].resource_id, "cpu-cold")
        self.assertEqual(profile.devices["gpu"].resource_id, "cuda0")
        self.assertEqual(
            profile.devices["phone"].resource_id, "op15-phone-compute"
        )
        task_profile = ProfileBundle.from_json({
            "schema": "s42-general-scheduler-profile-v1",
            "profile_id": "shared-calendar-task-profile",
            "resources": [
                {
                    "resource_id": resource.resource_id,
                    "kind": resource.kind,
                    "capacity": resource.capacity,
                    "ready": resource.ready,
                    "identity": resource.identity,
                }
                for resource in shared.values()
            ],
            "routes": [{
                "route_id": "cpu-task",
                "workload_id": "task",
                "granularity": "task",
                "baseline": True,
                "resource_slots": {"cpu-cold": 1},
                "latency": {
                    "cost_us": {
                        "fixed": 100,
                        "input_token": 0,
                        "kind": "affine_tokens_v1",
                        "output_token": 0,
                    },
                    "ucb_add_us": 0,
                    "sample_count": 1,
                    "measured": True,
                },
                "energy": {
                    "status": "unknown",
                    "cost_uj": None,
                    "lower_error_ppm": 0,
                    "upper_error_ppm": 0,
                },
                "overlap": {"status": "not_applicable"},
                "quality_class": "exact",
                "placement_verified": True,
                "resident": True,
                "server_busy_ppm": 1_000_000,
                "server_memory_bytes": 1,
                "evidence_ids": ["test-shared-calendar"],
            }],
            "trace_workload_map": {"task": "task"},
            "policy": {},
        })
        scheduler = UnifiedScheduler(
            (task_profile,), "control", matmul_profile=profile
        )
        self.assertEqual(
            set(scheduler.resource_snapshot(0)),
            {resource.resource_id for resource in shared.values()},
        )

    def test_invalid_memory_reservation_is_rejected(self) -> None:
        with self.assertRaises(MatmulScheduleError):
            materialize_matmul_profile(
                self.source,
                phone_capacity_bytes=1024,
                phone_reserved_bytes=1025,
            )


if __name__ == "__main__":
    unittest.main()
