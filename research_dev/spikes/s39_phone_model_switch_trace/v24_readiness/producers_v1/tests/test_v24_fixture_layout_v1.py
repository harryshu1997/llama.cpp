#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cuda = load_module("v24_fixture_cuda", "cuda_route_v1.py")
phone = load_module("v24_fixture_phone", "phone_route_v1.py")

PHASE_ID = "cp0-r1-v24-a-only-fixture"
BOOT_IDS = {
    "cuda": "11111111-1111-4111-8111-111111111111",
    "op12": "22222222-2222-4222-8222-222222222222",
    "op15": "33333333-3333-4333-8333-333333333333",
}


class V24FixtureLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name).resolve()
        self.artifact_dir = self.run / "artifact"
        self.pre_dir = self.run / "pre"
        self.fresh_dir = self.run / "fresh"
        for path in (self.artifact_dir, self.pre_dir, self.fresh_dir):
            path.mkdir()

        self.runtime = self.run / "cuda-route"
        self.runtime.write_bytes(b"runtime")
        runtime_stat = cuda.stat_record(os.stat(self.runtime))
        model_stat = dict(runtime_stat)
        model_stat["size"] = cuda.MODEL_BYTES
        self.components = [
            {
                "bytes": len(b"runtime"),
                "component_id": "cuda-route.launcher",
                "endpoint": "cuda",
                "kind": "runtime_component",
                "path": str(self.runtime),
                "sha256": cuda.sha256(b"runtime"),
                "stat": runtime_stat,
            },
            {
                "bytes": cuda.MODEL_BYTES,
                "component_id": "model.cuda",
                "endpoint": "cuda",
                "kind": "model_weight",
                "path": cuda.MODEL_PATH,
                "sha256": cuda.MODEL_SHA256,
                "stat": model_stat,
            },
        ]
        self.root_value = {
            "candidate_sha256": "1" * 64,
            "completed_ns": 2,
            "components": self.components,
            "contract_sha256": "2" * 64,
            "inventories": [{"bundle_id": "cuda_route"}],
            "model_id": cuda.MODEL_ID,
            "phase_scope": "PRE_REBOOT_OUTSIDE_PHASE",
            "runtime_bundle_plan_sha256": "3" * 64,
            "schema": "s39-cp0-r1-artifact-root-v2.4",
            "started_ns": 1,
        }
        self.root_raw = cuda.canonical_bytes(self.root_value)
        (self.artifact_dir / "artifact_root.json").write_bytes(self.root_raw)

        self.phase_value = {
            "artifact_root_sha256": cuda.sha256(self.root_raw),
            "candidate_sha256": "1" * 64,
            "contract_sha256": "2" * 64,
            "device_boot_ids": BOOT_IDS,
            "event_ns": 4,
            "model_id": cuda.MODEL_ID,
            "phase": "A_ONLY",
            "phase_id": PHASE_ID,
            "preparation_sha256": "4" * 64,
            "quality_corpus_sha256": "5" * 64,
            "runtime_bundle_plan_sha256": "3" * 64,
            "schema": "s39-cp0-r1-phase-lock-v2.4",
        }
        self.phase_raw = cuda.canonical_bytes(self.phase_value)
        (self.pre_dir / "phase_lock.jsonl").write_bytes(self.phase_raw)

        self.phone_plan = {
            "op12": {
                "boot_id": BOOT_IDS["op12"],
                "device": "OP595DL1",
                "model": "CPH2583",
                "product": "CPH2583",
                "serial": "5ae7a43d",
            },
            "op15": {
                "boot_id": BOOT_IDS["op15"],
                "device": "OP611FL1",
                "model": "CPH2749",
                "product": "CPH2749",
                "serial": "3C15AU002CL00000",
            },
        }
        devices = {
            "cuda": {
                "gpu_uuid": cuda.CUDA_UUID,
                "host": cuda.CUDA_HOST,
                "host_boot_id": BOOT_IDS["cuda"],
                "pci_bus_id": "0000:01:00.0",
                "system_swap_used_bytes": 0,
            },
        }
        for endpoint in ("op12", "op15"):
            devices[endpoint] = {
                **self.phone_plan[endpoint],
                "available_bytes": 1_000_000_000,
                "system_swap_used_bytes": 0,
                "thermal_status": 0,
            }
        self.fresh_value = {
            "artifact_root_sha256": cuda.sha256(self.root_raw),
            "component_stats": [
                {
                    "component_id": row["component_id"],
                    "endpoint": row["endpoint"],
                    "path": row["path"],
                    "stat": row["stat"],
                }
                for row in self.components
            ],
            "completed_ns": 6,
            "devices": devices,
            "inventories": [{"bundle_id": "cuda_route"}],
            "phase": "A_ONLY",
            "phase_id": PHASE_ID,
            "phase_lock_sha256": cuda.sha256(self.phase_raw),
            "preparation_sha256": "4" * 64,
            "runtime_bundle_plan_sha256": "3" * 64,
            "schema": "s39-cp0-r1-fast-fresh-readiness-v2.4",
            "started_ns": 5,
        }
        (self.fresh_dir / "fresh_snapshot.json").write_bytes(
            cuda.canonical_bytes(self.fresh_value)
        )

    def test_cuda_reads_exact_v24_artifact_and_fresh_paths(self):
        model, device, phase = cuda.load_base_evidence(
            self.pre_dir,
            PHASE_ID,
        )
        self.assertEqual(model["sha256"], cuda.MODEL_SHA256)
        self.assertEqual(device["gpu_uuid"], cuda.CUDA_UUID)
        self.assertEqual(
            phase["quality_corpus_sha256"],
            self.phase_value["quality_corpus_sha256"],
        )
        plan = {
            "worker": {
                "runtime_component_ids": ["cuda-route.launcher"],
                "runtime_executable": {
                    key: self.components[0][key]
                    for key in ("bytes", "path", "sha256", "stat")
                },
            },
        }
        components = cuda.load_runtime_evidence(
            self.pre_dir,
            PHASE_ID,
            plan,
        )
        self.assertEqual(set(components), {"cuda-route.launcher"})

    def test_phone_reads_v24_device_identity_without_stale_link_counters(self):
        phase, phase_raw = phone.load_phase(self.pre_dir, PHASE_ID)
        devices = phone.load_v24_fresh_devices(
            self.pre_dir,
            PHASE_ID,
            phase,
            phase_raw,
            {"phones": self.phone_plan},
        )
        self.assertEqual(devices["op12"]["boot_id"], BOOT_IDS["op12"])
        self.assertNotIn("interfaces", devices["op12"])

    def test_phone_runtime_counter_window_starts_at_inference_probe(self):
        expected = {
            **self.phone_plan["op15"],
            "direct_peer_ipv4": "10.0.0.2",
            "expected_worker_executable_path": "/data/local/tmp/worker",
            "interface": "wlan0",
            "loaded_shard_path": "/data/local/tmp/model.gguf",
            "local_ipv4": "10.0.0.1",
        }
        base = {
            "active_sequences": 0,
            "available_bytes": 1_000_000_000,
            "direct_peer": {
                "interface": "wlan0",
                "local_ipv4": "10.0.0.1",
                "peer_ipv4": "10.0.0.2",
                "socket_peer_observed": True,
            },
            "gpu_max_millic": 50_000,
            "interface": {
                "ipv4": "10.0.0.1",
                "name": "wlan0",
                "rx_bytes": 100,
                "tx_bytes": 200,
            },
            "process_swap_bytes": 0,
            "system_swap_used_bytes": 0,
            "worker_pid": 7,
            "worker_start_ticks": 8,
        }
        after = {
            **base,
            "interface": {
                **base["interface"],
                "rx_bytes": 300,
                "tx_bytes": 500,
            },
        }
        session = {
            "proto_version": 3,
            "worker_boot_nonce": 9,
            "worker_pid": 7,
        }
        value = phone.runtime_record(
            expected,
            base,
            after,
            session,
            11,
        )
        self.assertEqual(
            value["interface_before"],
            {"interface": "wlan0", "rx_bytes": 100, "tx_bytes": 200},
        )


if __name__ == "__main__":
    unittest.main()
