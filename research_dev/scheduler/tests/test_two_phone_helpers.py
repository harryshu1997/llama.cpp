"""Two FFN helper phones for one model server: manifests, transport, launch environment, accounting."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from research_dev.scheduler import GGUFModelManifestLoader
from research_dev.scheduler.adapters import llama_server_launch_contract
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler.adapters.llama_server_contracts import parse_llama_server_ffn_call
from research_dev.scheduler.adapters.phone_helpers import (
    PhoneHelperBinding,
    PhoneHelperTransportIdentity,
    attribute_ffn_calls,
    check_usb_topology,
    find_usb_device_by_serial,
    helper_layer_masks,
    helper_server_environment,
    helper_transport_contracts,
    layer_spec,
    phone_helper_bindings_from_parameters,
    validate_disjoint_ownership,
    validate_helper_identities,
)
from research_dev.scheduler.adapters.phone_tcp_session import (
    CONNECTED_MARKER,
    HELLO_REQUEST,
    HELLO_RESPONSE,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    AdbTcpPhoneWorkerSession,
    AdbTcpWorkerConfiguration,
)
from research_dev.scheduler.adapters.phone_transport import phone_transport_contract
from research_dev.scheduler.campaigns.burstgpt import launch
from research_dev.scheduler.config import load_scheduler_configuration
from research_dev.scheduler.configuration.common import SchedulerConfigurationError
from research_dev.scheduler.configuration.rig import RigManifest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from test_gguf_cost import write_synthetic_gguf  # noqa: E402
from test_llama_server_adapter import execution_command  # noqa: E402


SHA = "sha256:" + "a" * 64
OP15_SERIAL, PIXEL_SERIAL = "3C15AU002CL00000", "5A040DLCH004ES"
OP15_MASK = (1 << 18) - 1                 # Qwen layers 0-17 (HTP0..HTP2)
PIXEL_MASK = ((1 << 24) - 1) & ~OP15_MASK  # layers 18-23
REPO_ROOT = TESTS_DIR.parents[2]
BIN_DIR = Path(os.environ.get("S42_LLAMA_BUILD_BIN", REPO_ROOT / "build-cpu" / "bin"))

FUNCTIONFS_PARAMETERS = {
    "ffn_transport": "functionfs-usb", "usb_allocator": "devmem", "usb_batch_plan": "split-row",
    "usb_full_duplex": 1, "usb_max_payload_bytes": 40960, "usb_product_id": 0x2D00,
    "usb_queue_depth": 4, "usb_concurrent_streams": 4, "usb_slot_safety_bytes": 65536,
    "usb_split_h2d": 0, "usb_transport_generation": "functionfs-dmabuf-async-ring-v2",
    "usb_transport_profile_id": "op15-profile", "usb_vendor_id": 0x18D1, "usbfs_available_bytes": 1 << 30,
}
PIXEL_PARAMETERS = {
    "adb_port": 5037, "adb_serial": PIXEL_SERIAL, "ffn_transport": "adb-tcp",
    "ffn_worker_host": "127.0.0.1", "ffn_worker_port": 40317, "phone_worker_port": 26990,
}


def legacy_rig_json() -> dict:
    """The longtail dev2 desktop rig.json (read-only copy, paths need not exist)."""
    return {
        "binaries": {"adb": "/usr/bin/adb", "bridge": "/deploy/bridge", "close_helper": "/deploy/close.py",
                     "resident_server": "/deploy/cuda-build/bin/llama-server",
                     "server": "/deploy/cuda-build/bin/llama-server"},
        "devices": [
            {"device_id": "desktop-cpu", "kind": "cpu", "memory_capacity_bytes": 120000000000},
            {"device_id": "desktop-cuda", "kind": "gpu", "memory_capacity_bytes": 16000000000},
            {"device_id": "op15-phone", "kind": "phone", "memory_capacity_bytes": 10000000000},
        ],
        "endpoints": {"qwen_desktop": "http://127.0.0.1:18571", "qwen_phone": "http://127.0.0.1:18572"},
        "library_directories": {"cuda": "/deps/cuda/lib", "resident": "/deploy/cuda-build/bin"},
        "phone": {
            "adb_port": 5037, "android_gadget_path": "/config/usb_gadget/g1", "battery_ppm": 1000000,
            "boot_image_sha256": "sha256:f13c7c033ce74f32ea4aa6349cb531704407708f5fb35756a6c361ffb48322a3",
            "busybox_path": "/data/adb/magisk/busybox", "diagnostic_endpoint": "http://192.168.42.1:18383",
            "functionfs_gadget_path": "/config/usb_gadget/g2", "functionfs_root_path": "/dev/usb-ffs/s41",
            "kernel_release": "6.12.23-android16-5-o-g227664cbe007-4k", "minimum_usb_speed_mbps": 5000,
            "multi_session_port_base": 26760, "remote_hash_cache_path": "/deploy/PHONE_HASH_CACHE.json",
            "resident_router_path": "/data/local/tmp/router/llama-ffn-split-resident-router",
            "resident_workers_path": "/data/local/tmp/workers/llama-ffn-split-resident-workers",
            "restore_script": "/data/local/tmp/s42/restore_android_usb.sh", "serial": OP15_SERIAL,
            "session_root": "/data/local/tmp/s42-session",
            "session_script": "/data/local/tmp/s42/direct_phone_ffn_session.sh", "usb_controller": "a600000.dwc3",
            "whole_executable_device": "GPUOpenCL", "whole_forward_port": 29382,
            "whole_library_directory": "/data/local/tmp/llama-ubatch-op15/bin", "whole_remote_port": 18382,
            "whole_server_path": "/data/local/tmp/llama-ubatch-op15/bin/llama-server",
            "whole_server_sha256": "sha256:3fad5e2f2730d1e240f176010994c49dc4a6ea59fb970708df8d2a85bb0abe1c",
            "whole_state_directory": "/data/local/tmp/whole",
            "worker_path": "/data/local/tmp/bin/llama-ffn-split-worker",
        },
        "repo_root": "/deploy/source",
        "resources": [
            {"capacity": 8, "identity": "cuda0", "resource_id": "cuda0"},
            {"capacity": 4, "identity": "desktop-cpu", "resource_id": "desktop-cpu"},
            {"capacity": 1, "identity": "usb-root", "resource_id": "desktop-usb-root"},
            {"capacity": 1, "identity": "op15-adreno", "resource_id": "op15-adreno"},
            {"capacity": 1, "identity": "op15-functionfs", "resource_id": "op15-functionfs"},
            {"capacity": 1, "identity": "op15-htp", "resource_id": "op15-htp"},
            {"capacity": 1, "identity": "op15-ncm", "resource_id": "op15-ncm"},
        ],
        "rig_id": "physical-4060ti-op15-dormant-release-trace-v1",
        "schema": "research-scheduler-rig-v1",
        "topology": {
            "cpu_device_id": "desktop-cpu", "cpu_resource_id": "desktop-cpu",
            "functionfs_resource_id": "op15-functionfs", "gpu_device_id": "desktop-cuda",
            "gpu_exclusive_residency_resource_id": "cuda0", "gpu_memory_resource_id": "cuda0-vram",
            "gpu_resource_id": "cuda0", "host_memory_resource_id": "host-ram",
            "phone_compute_resource_ids": ["op15-htp"], "phone_device_id": "op15-phone",
            "phone_exclusive_residency_resource_id": "op15-htp", "phone_memory_resource_id": "op15-ram",
            "phone_transport_resource_ids": ["desktop-usb-root", "op15-functionfs", "op15-ncm"],
        },
        "transport_host_dependencies": {"ggml": "/deploy/cuda-build/bin/libggml.so"},
    }


def pixel_helper_json() -> dict:
    """The qualified Pixel 10 Pro Vulkan worker of reports/20260922-fast-path-M3 (gemv-server-1)."""
    return {
        "adb_port": 5037, "backend": "Vulkan0", "column_quantum": 4352, "device_id": "pixel10pro-phone",
        "forward_port": 0, "kernel_release": "6.6.102-android15-8-g6eb5b2a8c46b-ab14739656-4k",
        "library_directories": ["/data/local/tmp/s42-pixel10pro-gemv-tune-20260923-confirm-v1",
                                "/data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1"],
        "max_requests": 0, "max_tokens": 4, "minimum_usb_speed_mbps": 5000, "serial": PIXEL_SERIAL,
        "transport": "adb-tcp", "usb_sysfs_device": "2-9.2",
        "worker_environment": {"S42_PIXEL_F16_ROWS": "8", "S42_PIXEL_F16_SUBGROUP": "128", "S42_PIXEL_F16_WG": "128"},
        "worker_path": "/data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1/llama-ffn-split-worker",
        "worker_port": 26990,
    }


def two_phone_rig_json() -> dict:
    row = legacy_rig_json()
    row["helper_phones"] = [pixel_helper_json()]
    row["devices"].append({"device_id": "pixel10pro-phone", "kind": "phone", "memory_capacity_bytes": 15000000000})
    row["resources"] += [{"capacity": 1, "identity": "pixel10pro-gpu", "resource_id": "pixel10pro-gpu"},
                         {"capacity": 1, "identity": "pixel10pro-adb", "resource_id": "pixel10pro-adb"}]
    row["topology"]["helper_phones"] = [{
        "compute_resource_ids": ["pixel10pro-gpu"], "device_id": "pixel10pro-phone",
        "memory_resource_id": "pixel10pro-ram", "transport_resource_ids": ["desktop-usb-root", "pixel10pro-adb"],
    }]
    return row


def binding(label: str, device_id: str, serial: str, mask: int, parameters: dict | None = None) -> PhoneHelperBinding:
    return PhoneHelperBinding(device_id=device_id, serial=serial, layer_mask=mask, label=label,
                              transport_parameters=parameters or {})


OP15 = binding("op15", "op15-phone", OP15_SERIAL, OP15_MASK)
PIXEL = binding("pixel", "pixel10pro-phone", PIXEL_SERIAL, PIXEL_MASK, PIXEL_PARAMETERS)


class RigManifestTests(unittest.TestCase):
    def test_legacy_manifest_round_trips_byte_for_byte(self) -> None:
        row = legacy_rig_json()
        manifest = RigManifest.from_json(row, Path("/"))
        self.assertEqual(manifest.helper_phones, ())
        self.assertEqual(manifest.topology.helper_phones, ())
        self.assertEqual(json.dumps(manifest.to_json(), sort_keys=True), json.dumps(row, sort_keys=True))

    def test_helper_phone_binds_serial_transport_worker_and_resources(self) -> None:
        manifest = RigManifest.from_json(two_phone_rig_json(), Path("/"))
        pixel = manifest.helper_phone("pixel10pro-phone")
        self.assertEqual((pixel.serial, pixel.transport, pixel.backend, pixel.usb_sysfs_device),
                         (PIXEL_SERIAL, "adb-tcp", "Vulkan0", "2-9.2"))
        self.assertEqual(manifest.phone.serial, OP15_SERIAL)
        self.assertEqual(manifest.topology.helper_phones[0].compute_resource_ids, ("pixel10pro-gpu",))
        self.assertEqual(RigManifest.from_json(manifest.to_json(), Path("/")), manifest)

    def test_ambiguous_helper_phones_are_rejected(self) -> None:
        def mutate(change):
            row = two_phone_rig_json()
            change(row)
            return row
        cases = {
            "serial of the primary": lambda r: r["helper_phones"][0].update(serial=OP15_SERIAL),
            "device absent from topology": lambda r: r["topology"].update(helper_phones=[]),
            "device is the primary": lambda r: (r["helper_phones"][0].update(device_id="op15-phone"),
                                                r["topology"]["helper_phones"][0].update(device_id="op15-phone")),
            "shared compute": lambda r: r["topology"]["helper_phones"][0].update(compute_resource_ids=["op15-htp"]),
            "absent resource": lambda r: r["topology"]["helper_phones"][0].update(
                transport_resource_ids=["pixel10pro-ncm"]),
            "forward collides with whole-phone forward": lambda r: r["helper_phones"][0].update(forward_port=29382),
            "FunctionFS helper": lambda r: r["helper_phones"][0].update(transport="functionfs-usb"),
            "tensor bridge helper": lambda r: r["helper_phones"][0].update(transport="tcp"),
            "colon in a library directory": lambda r: r["helper_phones"][0].update(
                library_directories=["/data/a:/data/b"]),
            "library path in the environment": lambda r: r["helper_phones"][0]["worker_environment"].update(
                LD_LIBRARY_PATH="/x"),
            "quantum off the 32-column grid": lambda r: r["helper_phones"][0].update(column_quantum=4350),
            "device kind": lambda r: r["devices"][-1].update(kind="gpu"),
        }
        for name, change in cases.items():
            with self.subTest(case=name), self.assertRaises(SchedulerConfigurationError):
                RigManifest.from_json(mutate(change), Path("/"))


class CampaignConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _write(self, rig: dict, helper_shards: dict | None) -> Path:
        model = {
            "backend_ids": {"desktop": "llama-server-cuda-cpu", "phone": "llama-server-cuda-cpu-op15"},
            "calibration_key": "qwen", "checked_manifest_path": "QWEN_MANIFEST.json",
            "endpoint_ids": {"desktop": "qwen_desktop", "phone": "qwen_phone"},
            "host_artifact_path": "qwen.gguf", "kind": "assisted", "model_id": "qwen3-14b-q4km-dequant-f16",
            "model_key": "hot", "phone_artifact_path": "/data/local/tmp/qwen.gguf", "trace_role": "hot",
            "transition_load_metric": "hot_load_ms", "transition_warm_metric": "hot_warm_ms",
        }
        if helper_shards is not None:
            model["helper_phone_ffn_shards"] = helper_shards
        cold = {**model, "calibration_key": "gemma", "model_id": "gemma-4-12b", "model_key": "cold",
                "trace_role": "cold", "host_artifact_path": "gemma.gguf"}
        cold.pop("helper_phone_ffn_shards", None)
        files = {
            "rig.json": rig,
            "models.json": {"desktop_baseline_plans_path": "plans.json", "manifest_cache_path": "cache.json",
                            "models": [model, cold, {
                                "backend_ids": {"desktop": "llama-server-cuda-cpu"},
                                "endpoint_ids": {"desktop": "qwen_desktop"}, "host_artifact_path": "llama.gguf",
                                "kind": "overlay", "model_id": "llama-1b", "model_key": "overlay",
                                "phone_artifact_path": "/data/local/tmp/llama.gguf", "trace_role": "overlay"}],
                            "schema": "research-scheduler-models-v1"},
            "evidence.json": {
                "adaptive_observation_source_catalog_path": "a.json", "adaptive_observation_store_path": "b.json",
                "assisted_calibration_arm": "op15", "calibration_directory": "calibration",
                "calibration_runs": {"control": ["c1"], "op15": ["o1"]}, "desktop_calibration_arm": "control",
                "kernel_profile_path": "kernel.json", "observation_source_catalog_path": "c.json",
                "observation_store_path": "d.json", "overlay_catalog_path": "e.json",
                "phone_power": {"active_power_mw": 4500, "allow_assumed_for_scheduling": True,
                                "evidence_kind": "ASSUMED_4P5W", "idle_power_mw": 875, "minimum_battery_ppm": 50000},
                "phone_session_discovery_path": "f.json", "schema": "research-scheduler-evidence-v1",
                "transport_qualification_directories": ["receipts"],
                "transport_qualification_identity_path": "identity.json",
            },
            "campaign.json": {
                "campaign_id": "two-phone-config-test", "energy_attribution_kind": "diagnostic",
                "evidence_manifest_path": "evidence.json", "maximum_latency_ppm": 1250000,
                "models_manifest_path": "models.json", "rig_manifest_path": "rig.json",
                "schema": "research-scheduler-campaign-v1", "selection_mode": "energy-aware",
                "trace": {"large_requests_path": "large.jsonl", "overlay_requests_path": "overlay.jsonl",
                          "trace_manifest_path": "trace.json"},
            },
        }
        for name, value in files.items():
            (self.root / name).write_text(json.dumps(value), encoding="ascii")
        return self.root / "campaign.json"

    def test_helper_shards_bind_to_rig_helper_phones(self) -> None:
        shards = {"pixel10pro-phone": {"index_path": "pixel/FFN_SHARDS.json",
                                       "directory": "/data/local/tmp/s42-pixel10pro-qualification-20260922-v1"}}
        configuration = load_scheduler_configuration(self._write(two_phone_rig_json(), shards), environ={})
        model = configuration.models.models[0]
        self.assertEqual(model.helper_phone_ffn_shards["pixel10pro-phone"],
                         (self.root / "pixel/FFN_SHARDS.json", shards["pixel10pro-phone"]["directory"]))
        self.assertEqual(model.to_json()["helper_phone_ffn_shards"]["pixel10pro-phone"]["directory"],
                         shards["pixel10pro-phone"]["directory"])
        command = launch.preflight_command(configuration, catalog_path=self.root / "c.json",
                                           normal_usb_receipt_path=self.root / "u.json",
                                           output_path=self.root / "out.json")
        rows = [json.loads(command[index + 1]) for index, value in enumerate(command) if value == "--helper-phone"]
        self.assertEqual(rows, [pixel_helper_json()])
        with self.assertRaises(SchedulerConfigurationError):
            load_scheduler_configuration(self._write(legacy_rig_json(), shards), environ={})

    def test_legacy_campaign_is_unchanged_and_two_phone_run_fails_closed(self) -> None:
        legacy = load_scheduler_configuration(self._write(legacy_rig_json(), None), environ={})
        self.assertNotIn("helper_phone_ffn_shards", legacy.models.models[0].to_json())
        command = launch.preflight_command(legacy, catalog_path=self.root / "c.json",
                                           normal_usb_receipt_path=self.root / "u.json",
                                           output_path=self.root / "out.json")
        self.assertNotIn("--helper-phone", command)
        campaign = self._write(two_phone_rig_json(), None)
        with self.assertRaisesRegex(Exception, "qualified evidence and a campaign lifecycle"):
            launch.launch(campaign, self.root / "run-output", preflight_only=False, resolve_only=False)
        self.assertFalse((self.root / "run-output").exists())


class TransportContractTests(unittest.TestCase):
    def test_adb_tcp_is_a_direct_worker_the_server_sees_as_tcp(self) -> None:
        contract = phone_transport_contract(PIXEL_PARAMETERS)
        self.assertEqual((contract.transport, contract.adb_serial, contract.phone_worker_port, contract.batch_plan),
                         ("adb-tcp", PIXEL_SERIAL, 26990, "single"))
        self.assertFalse(contract.uses_tensor_bridge)
        self.assertEqual(dict(contract.server_environment()), {
            "S41_SERVER_FFN_HOST": "127.0.0.1", "S41_SERVER_FFN_PORT": "40317", "S41_SERVER_FFN_TRANSPORT": "tcp"})
        for key in ("adb_serial", "phone_worker_port", "ffn_worker_port"):
            with self.subTest(missing=key), self.assertRaises(PhysicalAdapterError):
                phone_transport_contract({k: v for k, v in PIXEL_PARAMETERS.items() if k != key})

    def test_legacy_transports_are_unchanged(self) -> None:
        bridge = phone_transport_contract({"ffn_transport": "tcp", "bridge_allocator": "devmem",
                                           "ffn_bridge_host": "127.0.0.1", "ffn_bridge_port": 9000,
                                           "bridge_queue_depth": 2})
        self.assertTrue(bridge.uses_tensor_bridge)
        self.assertIsNone(bridge.adb_serial)
        usb = phone_transport_contract(FUNCTIONFS_PARAMETERS)
        self.assertEqual(usb.server_environment()["S41_SERVER_FFN_TRANSPORT"], "functionfs-usb")
        self.assertIsNone(usb.adb_serial)


class OwnershipAndEnvironmentTests(unittest.TestCase):
    SHARED = {"S41_SERVER_FFN_ARTIFACT_SHA256": SHA, "S41_SERVER_FFN_COLUMNS": "17408",
              "S41_SERVER_FFN_LAYER_MASK": str(OP15_MASK | PIXEL_MASK), "S41_SERVER_FFN_RUNTIME_CONTROL": "1",
              "S41_SERVER_FFN_DORMANT_HOST_SHARE": "1"}

    def test_disjoint_ownership_and_policy_sub_masks(self) -> None:
        self.assertEqual(validate_disjoint_ownership((OP15, PIXEL)), (1 << 24) - 1)
        self.assertEqual(layer_spec(PIXEL_MASK), "18-23")
        self.assertEqual(layer_spec(0b1011), "0-1,3")
        policy = (1 << 12) - 1 | 1 << 20
        self.assertEqual(dict(helper_layer_masks(policy, (OP15, PIXEL))),
                         {"op15-phone": (1 << 12) - 1, "pixel10pro-phone": 1 << 20})
        with self.assertRaises(PhysicalAdapterError):
            helper_layer_masks(1 << 30, (OP15, PIXEL))
        for helpers, required in (
            ((OP15, replace(PIXEL, layer_mask=PIXEL_MASK | 1)), None),                  # overlap
            ((OP15, replace(PIXEL, serial=OP15_SERIAL)), None),                         # duplicate serial
            ((OP15, replace(PIXEL, label="op15")), None),                               # duplicate label
            ((OP15, PIXEL), (1 << 25) - 1),                                              # layer 24 uncovered
        ):
            with self.subTest(helpers=helpers), self.assertRaises(PhysicalAdapterError):
                validate_disjoint_ownership(helpers, required_layer_mask=required)

    def test_one_helper_is_the_legacy_environment(self) -> None:
        contracts = helper_transport_contracts((OP15,), FUNCTIONFS_PARAMETERS)
        shared = {**self.SHARED, "S41_SERVER_FFN_LAYER_MASK": str(OP15_MASK)}
        legacy = {**shared, **phone_transport_contract(FUNCTIONFS_PARAMETERS).server_environment()}
        self.assertEqual(dict(helper_server_environment((OP15,), shared, contracts)), dict(sorted(legacy.items())))

    def test_two_helpers_get_prefixed_transports_and_masks(self) -> None:
        contracts = helper_transport_contracts((OP15, PIXEL), FUNCTIONFS_PARAMETERS)
        environment = dict(helper_server_environment((OP15, PIXEL), self.SHARED, contracts))
        self.assertEqual(environment["S41_SERVER_FFN_HELPERS"], "2")
        self.assertEqual(environment["S41_SERVER_FFN_HELPER0_LABEL"], "op15")
        self.assertEqual(environment["S41_SERVER_FFN_HELPER0_LAYER_MASK"], str(OP15_MASK))
        self.assertEqual(environment["S41_SERVER_FFN_HELPER0_TRANSPORT"], "functionfs-usb")
        self.assertEqual(environment["S41_SERVER_FFN_HELPER0_USB_VENDOR_ID"], str(0x18D1))
        self.assertEqual(environment["S41_SERVER_FFN_HELPER1_TRANSPORT"], "tcp")
        self.assertEqual(environment["S41_SERVER_FFN_HELPER1_PORT"], "40317")
        self.assertEqual(environment["S41_SERVER_FFN_HELPER1_LAYER_MASK"], str(PIXEL_MASK))
        self.assertFalse({"S41_SERVER_FFN_TRANSPORT", "S41_SERVER_FFN_HOST", "S41_SERVER_FFN_PORT"} & set(environment))
        self.assertTrue({key: value for key, value in self.SHARED.items()}.items() <= environment.items())

    def test_ambiguous_environments_fail_closed(self) -> None:
        second_usb = binding("op15b", "op15b-phone", "OTHER", PIXEL_MASK, FUNCTIONFS_PARAMETERS)
        bridge = binding("bridge", "bridge-phone", "BRIDGE", PIXEL_MASK, {
            "ffn_transport": "tcp", "bridge_allocator": "devmem", "ffn_bridge_host": "127.0.0.1",
            "ffn_bridge_port": 9000, "bridge_queue_depth": 2})
        wrong_serial = replace(PIXEL, serial="SOMEONE-ELSE")
        for helpers in ((OP15, second_usb), (OP15, bridge), (OP15, wrong_serial),
                        (OP15, replace(PIXEL, transport_parameters={}))):
            with self.subTest(helpers=[row.label for row in helpers]), self.assertRaises(PhysicalAdapterError):
                helper_transport_contracts(helpers, FUNCTIONFS_PARAMETERS)
        contracts = helper_transport_contracts((OP15, PIXEL), FUNCTIONFS_PARAMETERS)
        for extra in ({"S41_SERVER_FFN_ROW_DIAGNOSTIC_STEPS": "5"}, {"S41_SERVER_FFN_HOST": "x"},
                      {"S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK": "1"},
                      {"S41_SERVER_FFN_LAYER_MASK": str(OP15_MASK)}):
            with self.subTest(extra=extra), self.assertRaises(PhysicalAdapterError):
                helper_server_environment((OP15, PIXEL), {**self.SHARED, **extra}, contracts)


class LaunchContractTests(unittest.TestCase):
    """``phone_helpers`` on a ticket turns its single-phone environment into one client per phone."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        model = Path(self.directory.name) / "model.gguf"
        write_synthetic_gguf(model, block_count=2)
        self.manifest = GGUFModelManifestLoader.load("synthetic-model-id", model)
        command = execution_command(self.manifest.artifact_sha256)
        operators = list(command.operator_plan["operators"]) + [
            {**row, "operator_id": row["operator_id"].replace("layer:0", "layer:1")}
            for row in command.operator_plan["operators"]
        ]
        self.command = replace(command, operator_plan={**command.operator_plan, "operators": operators})

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _command(self, helpers: list[dict] | None):
        parameters = dict(self.command.adapter_parameters)
        if helpers is not None:
            parameters["phone_helpers"] = json.dumps(helpers)
        return replace(self.command, adapter_parameters=parameters)

    def test_ticket_without_helpers_keeps_its_environment(self) -> None:
        environment = llama_server_launch_contract(self._command(None), self.manifest).ffn_environment
        self.assertEqual(environment["S41_SERVER_FFN_LAYER_MASK"], "3")
        self.assertEqual(environment["S41_SERVER_FFN_TRANSPORT"], "functionfs-usb")
        self.assertNotIn("S41_SERVER_FFN_HELPERS", environment)

    def test_ticket_with_two_helpers_launches_one_client_per_phone(self) -> None:
        helpers = [
            {"device_id": "helper-c", "label": "op15", "layer_mask": 1, "serial": OP15_SERIAL},
            {"device_id": "pixel10pro-phone", "label": "pixel", "layer_mask": 2, "serial": PIXEL_SERIAL,
             "transport_parameters": PIXEL_PARAMETERS},
        ]
        environment = llama_server_launch_contract(self._command(helpers), self.manifest).ffn_environment
        self.assertEqual(environment["S41_SERVER_FFN_HELPERS"], "2")
        self.assertEqual(environment["S41_SERVER_FFN_LAYER_MASK"], "3")
        self.assertEqual(environment["S41_SERVER_FFN_HELPER0_TRANSPORT"], "functionfs-usb")
        self.assertEqual(environment["S41_SERVER_FFN_HELPER1_PORT"], "40317")
        self.assertNotIn("S41_SERVER_FFN_TRANSPORT", environment)
        self.assertIn("S41_SERVER_FFN_POLICY", environment)
        bad = copy.deepcopy(helpers)
        bad[1]["layer_mask"] = 3
        for helpers_row in (bad, list(reversed(helpers)), helpers[:1] + [{**helpers[1], "layer_mask": 4}]):
            with self.subTest(helpers=helpers_row), self.assertRaises(PhysicalAdapterError):
                llama_server_launch_contract(self._command(helpers_row), self.manifest)
        self.assertIsNone(phone_helper_bindings_from_parameters({"phone_device_id": "x"}))


class CallAttributionTests(unittest.TestCase):
    def _calls(self, rows):
        return [parse_llama_server_ffn_call(
            f"S41SERVERFFNCALL request={request} layer={layer} tokens={tokens} columns=13056 "
            f"payload_bytes={tokens * 10240}") for request, layer, tokens in rows]

    def test_calls_are_counted_per_owning_phone(self) -> None:
        second = 1 + (1 << 24)
        calls = self._calls([(1, 0, 1), (2, 17, 1), (second, 18, 1), (second + 1, 23, 2), (3, 5, 2)])
        accounts = attribute_ffn_calls(calls, (OP15, PIXEL))
        self.assertEqual({key: (row.calls, row.rows, row.layers_seen) for key, row in accounts.items()},
                         {"op15-phone": (3, 4, (0, 5, 17)), "pixel10pro-phone": (2, 3, (18, 23))})
        self.assertEqual(accounts["pixel10pro-phone"].first_request_id, second)
        self.assertEqual(accounts["pixel10pro-phone"].to_json()["payload_bytes"], 3 * 10240)

    def test_unowned_repeated_or_misrouted_calls_fail_closed(self) -> None:
        for rows in ([(1, 30, 1)], [(1, 0, 1), (1, 1, 1)], [(5, 18, 1)], [(1 + (1 << 24), 3, 1)]):
            with self.subTest(rows=rows), self.assertRaises(PhysicalAdapterError):
                attribute_ffn_calls(self._calls(rows), (OP15, PIXEL))


class TransportIdentityTests(unittest.TestCase):
    PIXEL_IDENTITY = {
        "device_id": "pixel10pro-phone", "transport": "adb-tcp", "transport_generation": "adb-tcp-worker-v6",
        "minimum_usb_speed_mbps": 5000,
        "hardware_identity": {"adb_usb_identity": "18d1:4ee7", "host_usb_controller": "0000:00:14.0",
                              "phone_kernel_release": "6.6.102-android15-8-g6eb5b2a8c46b-ab14739656-4k",
                              "phone_usb_serial": PIXEL_SERIAL, "phone_usb_sysfs_device": "2-9.2"},
        "software_identity": {"host_binary_sha256": SHA, "phone_shard_sha256": SHA, "phone_worker_sha256": SHA,
                              "transport_client_source_sha256": SHA, "worker_environment_sha256": SHA,
                              "phone_library_sha256:libggml-vulkan.so": SHA},
        "receipts": {"numerical-rows-1-2-4": SHA, "server-token-identity": SHA, "usb-link-speed": SHA},
    }

    def test_missing_receipts_are_reported_and_block_the_helper(self) -> None:
        identity = PhoneHelperTransportIdentity(**self.PIXEL_IDENTITY)
        self.assertEqual(identity.missing_receipts, ("adb-forward-round-trip", "scheduler-launched-session"))
        self.assertFalse(identity.qualified)
        self.assertEqual(PhoneHelperTransportIdentity.from_json(identity.to_json()), identity)
        pixel_only = replace(PIXEL, transport_parameters=PIXEL_PARAMETERS)
        observed = {"pixel10pro-phone": type("Observed", (), {"serial": PIXEL_SERIAL, "sysfs_device": "2-9.2",
                                                             "negotiated_speed_mbps": 5000})()}
        with self.assertRaisesRegex(PhysicalAdapterError, "lacks receipts"):
            validate_helper_identities((pixel_only,), {"pixel10pro-phone": identity}, observed)
        complete = replace(identity, receipts={**identity.receipts, "adb-forward-round-trip": SHA,
                                               "scheduler-launched-session": SHA})
        validate_helper_identities((pixel_only,), {"pixel10pro-phone": complete}, observed)
        moved = {"pixel10pro-phone": type("Observed", (), {"serial": PIXEL_SERIAL, "sysfs_device": "2-10",
                                                          "negotiated_speed_mbps": 5000})()}
        with self.assertRaisesRegex(PhysicalAdapterError, "USB link differs"):
            validate_helper_identities((pixel_only,), {"pixel10pro-phone": complete}, moved)
        with self.assertRaisesRegex(PhysicalAdapterError, "no transport identity of its own"):
            validate_helper_identities((OP15, pixel_only), {"pixel10pro-phone": complete}, observed)

    def test_identity_requires_its_hardware_and_library_hashes(self) -> None:
        for change in ({"hardware_identity": {"phone_usb_serial": PIXEL_SERIAL}},
                       {"software_identity": {**self.PIXEL_IDENTITY["software_identity"],
                                              "phone_library_sha256:libggml-vulkan.so": "abc"}},
                       {"transport_generation": "functionfs-dmabuf-async-ring-v2"},
                       {"receipts": {"usb-link-speed": "not-a-hash"}}):
            with self.subTest(change=list(change)), self.assertRaises(PhysicalAdapterError):
                PhoneHelperTransportIdentity(**{**self.PIXEL_IDENTITY, **change})


class UsbTopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.sysfs = root / "bus"
        self.sysfs.mkdir()
        # the desktop on 2026-09-24: OP15 on root port 2, Pixel behind the ASM107x hub on root port 9
        for name, serial, ids in (("2-2", OP15_SERIAL, ("22d9", "2772")), ("2-9.2", PIXEL_SERIAL, ("18d1", "4ee7")),
                                  ("2-9.3", "SECOND", ("18d1", "4ee7"))):
            device = root / "devices/pci0000:00/0000:00:14.0/usb2" / name
            device.mkdir(parents=True)
            for field, value in (("serial", serial), ("idVendor", ids[0]), ("idProduct", ids[1]), ("speed", "5000")):
                (device / field).write_text(value + "\n")
            (self.sysfs / name).symlink_to(device)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_desktop_topology_passes_and_reports_the_shared_controller(self) -> None:
        self.assertEqual(find_usb_device_by_serial(OP15_SERIAL, sysfs_root=self.sysfs), "2-2")
        checks, observed = check_usb_topology(
            [("op15-phone", OP15_SERIAL, None, 5000), ("pixel10pro-phone", PIXEL_SERIAL, "2-9.2", 5000)],
            sysfs_root=self.sysfs)
        self.assertTrue(all(row.passed for row in checks), [row.to_json() for row in checks])
        self.assertEqual(observed["pixel10pro-phone"].root_port, "2-9")
        self.assertTrue(observed["pixel10pro-phone"].behind_hub)
        self.assertIn("controllers 0000:00:14.0; root ports 2-2,2-9", checks[-1].detail)

    def test_wrong_serial_speed_or_shared_hub_uplink_fails(self) -> None:
        for phones in (
            [("pixel10pro-phone", "OTHER", "2-9.2", 5000)],
            [("pixel10pro-phone", PIXEL_SERIAL, "2-9.2", 10000)],
            [("pixel10pro-phone", PIXEL_SERIAL, "2-9.2", 5000), ("second", "SECOND", "2-9.3", 5000)],
            [("absent", "NOPE", None, 5000)],
        ):
            with self.subTest(phones=phones):
                checks, _ = check_usb_topology(phones, sysfs_root=self.sysfs)
                self.assertFalse(all(row.passed for row in checks))


FAKE_ADB = r'''#!/usr/bin/env python3
"""adb stand-in: runs phone shell commands on this host (no device, no forward)."""
import os, subprocess, sys
args = sys.argv[1:]
while args and args[0] in ("-P", "-s"):
    args = args[2:]
if args[:2] == ["shell", "-T"]:
    os.execvp("sh", ["sh", "-c", args[2]])
if args[:1] == ["shell"]:
    command = args[1].replace("ps -A -o PID,ARGS", "ps -A -o pid,args")
    sys.exit(subprocess.call(["sh", "-c", command]))
if args[:2] == ["forward", "--no-rebind"]:
    print(args[3].split(":")[1])
    sys.exit(0)
if args[:2] == ["forward", "--remove"]:
    sys.exit(0)
sys.exit(2)
'''


class AdbTcpSessionTests(unittest.TestCase):
    """The helper lifecycle against a real local CPU worker behind a fake adb."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.worker = BIN_DIR / "llama-ffn-split-worker"
        if not cls.worker.exists():
            raise unittest.SkipTest(f"llama-ffn-split-worker missing under {BIN_DIR}")
        from tiny_llama_gguf import N_EMBD, N_FF, write_tiny_llama_gguf
        cls.n_embd, cls.n_ff = N_EMBD, N_FF
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.model = write_tiny_llama_gguf(cls.root / "tiny.gguf", n_layer=4)
        cls.artifact = "sha256:" + subprocess.check_output(["sha256sum", str(cls.model)], text=True).split()[0]
        cls.adb = cls.root / "adb"
        cls.adb.write_text(FAKE_ADB)
        cls.adb.chmod(cls.adb.stat().st_mode | stat.S_IEXEC)
        cls.worker_copy = cls.root / "bin" / "llama-ffn-split-worker"
        cls.worker_copy.parent.mkdir()
        shutil.copy2(cls.worker, cls.worker_copy)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def _configuration(self, **overrides) -> AdbTcpWorkerConfiguration:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        digest = lambda path: "sha256:" + subprocess.check_output(["sha256sum", str(path)], text=True).split()[0]  # noqa: E731
        values = dict(
            device_id="pixel10pro-phone", serial=PIXEL_SERIAL, adb_port=5037, adb_path=self.adb,
            worker_path=str(self.worker_copy), library_directories=(str(BIN_DIR.resolve()),),
            shard_path=str(self.model), artifact_sha256=self.artifact, layer_mask=0b1100, n_embd=self.n_embd,
            columns=self.n_ff, column_quantum=64, max_tokens=4, swiglu=True, backend="CPU", phone_port=port,
            max_requests=7, worker_environment={"S42_TEST": "1"},
            expected_sha256_by_path={str(self.worker_copy): digest(self.worker_copy), str(self.model): self.artifact},
            launch_timeout_s=30.0)
        values.update(overrides)
        return AdbTcpWorkerConfiguration(**values)

    def test_finite_budget_session_starts_serves_and_drains_to_a_normal_exit(self) -> None:
        session = AdbTcpPhoneWorkerSession(self._configuration())
        preflight = session.preflight()
        self.assertTrue(preflight.details["port_free"])
        launch_receipt = session.start(self.root / "worker-finite.log")
        self.assertIn("layers=2 mask=000000000000000c", launch_receipt.details["ready_line"])
        self.assertEqual(session.transport_contract().transport, "adb-tcp")
        self.assertEqual(session.binding("pixel").layer_mask, 0b1100)
        # a first client (the server's role) makes two calls, then the stop drains the other five
        self.assertEqual(session._drain(5), 2)
        with self.assertRaisesRegex(PhysicalAdapterError, "served call count"):
            session.stop()
        stop = session.stop(served_calls=2)
        self.assertEqual((stop.details["exit_code"], stop.details["drained_calls"], stop.details["signalled"]),
                         (0, 5, False))
        self.assertTrue(stop.details["boot_unchanged"])
        self.assertEqual(stop.details["worker_pids_after"], [])

    def test_a_low_served_count_ends_the_drain_when_the_worker_exits(self) -> None:
        session = AdbTcpPhoneWorkerSession(self._configuration(max_requests=4))
        session.start(self.root / "worker-low-count.log")
        self.assertEqual(session._drain(2), 2)
        stop = session.stop(served_calls=0)
        self.assertEqual((stop.details["exit_code"], stop.details["drained_calls"]), (0, 2))

    def test_resident_worker_is_signalled_only_on_request_and_when_idle(self) -> None:
        session = AdbTcpPhoneWorkerSession(self._configuration(max_requests=0))
        session.start(self.root / "worker-resident.log")
        with self.assertRaisesRegex(PhysicalAdapterError, "stays running"):
            session.stop()
        stop = session.stop(allow_idle_signal=True)
        self.assertTrue(stop.details["signalled"])
        self.assertEqual(stop.details["worker_pids_after"], [])

    def test_resident_worker_killed_mid_call_is_released_only_on_request(self) -> None:
        """G1b with a real worker: SIGTERM while the server's client is connected. The worker never
        logs that the client left, so the plain stop still refuses it as busy (flag-absent behaviour);
        the elastic ``release_exited`` stop sees no worker left and removes the forward."""
        configuration = self._configuration(max_requests=0)
        session = AdbTcpPhoneWorkerSession(configuration)
        launch_receipt = session.start(self.root / "worker-killed.log")
        (pid,) = launch_receipt.details["worker_pids"]
        with socket.create_connection(("127.0.0.1", session.transport_parameters()["ffn_worker_port"])) as client:
            client.sendall(HELLO_REQUEST.pack(
                PROTOCOL_MAGIC, PROTOCOL_VERSION, 1, configuration.layer_mask, configuration.n_embd,
                configuration.columns, 1 | 2, configuration.max_tokens,
                bytes.fromhex(configuration.artifact_sha256[7:])))
            hello = HELLO_RESPONSE.unpack(session._receive(client, HELLO_RESPONSE.size))
            self.assertEqual(hello[:4], (PROTOCOL_MAGIC, PROTOCOL_VERSION, 2, 0))
            for _ in range(200):
                if CONNECTED_MARKER in session._log_path.read_text():
                    break
                time.sleep(0.05)
            os.kill(pid, signal.SIGTERM)
            session._process.wait(timeout=30)
        self.assertIn(CONNECTED_MARKER, session._log_path.read_text())
        with self.assertRaisesRegex(PhysicalAdapterError, "stays running"):
            session.stop(allow_idle_signal=True)
        stop = session.stop(allow_idle_signal=True, release_exited=True)
        self.assertEqual(
            {key: stop.details[key] for key in ("already_exited", "signalled", "forward_removed", "worker_pids_after")},
            {"already_exited": True, "signalled": False, "forward_removed": True, "worker_pids_after": []})
        self.assertEqual(stop.details["exit_code"], -signal.SIGTERM)
        self.assertFalse(session.active)

    def test_preflight_refuses_hash_mismatch_and_occupied_port(self) -> None:
        configuration = self._configuration()
        bad = replace(configuration, expected_sha256_by_path={
            **configuration.expected_sha256_by_path, str(self.model): SHA})
        with self.assertRaisesRegex(PhysicalAdapterError, "differ from their pinned sha256"):
            AdbTcpPhoneWorkerSession(bad).preflight()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", configuration.phone_port))
            listener.listen(1)
            with self.assertRaisesRegex(PhysicalAdapterError, "is occupied"):
                AdbTcpPhoneWorkerSession(configuration).preflight()

    def test_configuration_pins_worker_and_shard(self) -> None:
        configuration = self._configuration()
        command = configuration.worker_command()
        self.assertEqual(command[:3], ("env", "LD_LIBRARY_PATH=" + str(BIN_DIR.resolve()), "S42_TEST=1"))
        self.assertIn("--layers", command)
        self.assertEqual(command[command.index("--layers") + 1], "2-3")
        for change in ({"expected_sha256_by_path": {}}, {"library_directories": ()},
                       {"worker_environment": {"LD_LIBRARY_PATH": "/x"}}, {"columns": 100}):
            with self.subTest(change=list(change)), self.assertRaises(PhysicalAdapterError):
                replace(configuration, **change)

    def test_optional_root_launch_preserves_the_phone_lock_and_legacy_command(self) -> None:
        configuration = self._configuration()
        self.assertEqual(configuration.worker_shell_command(), "exec " + shlex.join(configuration.worker_command()))
        rooted = replace(configuration, as_root=True, phone_lock_path="/data/local/tmp/pixel.lock")
        words = shlex.split(rooted.worker_shell_command())
        self.assertEqual(words[:2], ["su", "-c"])
        self.assertEqual(len(words), 3)
        self.assertTrue(words[2].startswith("exec 9>/data/local/tmp/pixel.lock\nflock -n 9 9>&9 || exit 73\n"))
        self.assertTrue(words[2].endswith("exec " + shlex.join(configuration.worker_command())))
        for change in ({"as_root": "yes"}, {"phone_lock_path": "relative"},
                       {"phone_lock_path": "/data/local/tmp/bad path"}):
            with self.subTest(change=change), self.assertRaises(PhysicalAdapterError):
                replace(configuration, **change)

    def test_rooted_basename_pid_requires_the_exact_executable(self) -> None:
        configuration = replace(self._configuration(), as_root=True)
        session = AdbTcpPhoneWorkerSession(configuration)
        listing = "PID ARGS\n123 llama-ffn-split-worker -m shard\n124 llama-ffn-split-worker -m other\n125 sh -c worker\n"
        with patch.object(session, "_shell", side_effect=[listing, configuration.worker_path, "/other/llama-ffn-split-worker"]):
            self.assertEqual(session._worker_pids(), [123])


class _Json:
    def __init__(self, value):
        self.value = value

    def to_json(self):
        return self.value


class FakeOp15Owner:
    def __init__(self):
        self.proofs = {}
        self.events = []

    def preflight(self):
        self.events.append("preflight")
        return _Json({"owner": "op15"})

    def start(self, command, manifest, usb):
        self.events.append("start")
        return _Json({"ready": True})

    def bind_ticket_generation(self, ticket_id):
        return 1

    def record_execution_proof(self, ticket_id, artifact, proofs):
        self.proofs[ticket_id] = [row.session_id for row in proofs]

    def finish(self, *, require_execution):
        self.events.append("finish")
        return _Json({"closed": True})

    def abort(self):
        return _Json({"aborted": True})


class FakeHelperSession:
    instances = []

    def __init__(self, configuration):
        self.configuration = configuration
        self.stopped_with = None
        FakeHelperSession.instances.append(self)

    def preflight(self):
        return _Json({"helper": "preflight"})

    def start(self, log_path):
        return _Json({"helper": "launch"})

    def binding(self, label):
        return PhoneHelperBinding(self.configuration.device_id, self.configuration.serial,
                                  self.configuration.layer_mask, label, PIXEL_PARAMETERS)

    def transport_contract(self):
        return phone_transport_contract(PIXEL_PARAMETERS)

    def stop(self, *, served_calls=None):
        self.stopped_with = served_calls
        return _Json({"exit_code": 0, "drained_calls": 100 - served_calls})


class TwoPhoneGateTests(unittest.TestCase):
    """The M3 gate wrapper around the decode-relocation gate, without phones."""

    def setUp(self) -> None:
        from research_dev.scheduler.campaigns.burstgpt import two_phone_gate
        self.module = two_phone_gate
        self.directory = tempfile.TemporaryDirectory()
        self.output = Path(self.directory.name)
        self.manifest = argparse.Namespace(
            block_count=40, feed_forward_length=17408, embedding_length=5120, artifact_sha256=SHA,
            tensor_by_id={f"blk.{il}.ffn_{kind}.weight": argparse.Namespace(nbytes=178257920)
                          for il in range(40) for kind in ("gate", "up", "down")})
        self.usb = phone_transport_contract(FUNCTIONFS_PARAMETERS)
        self.op15 = FakeOp15Owner()
        env = {"S41_SERVER_FFN_ACTIVATION": "swiglu", "S41_SERVER_FFN_ARTIFACT_SHA256": SHA,
               "S41_SERVER_FFN_COLUMNS": "17408", "S41_SERVER_FFN_LAYER_MASK": str(OP15_MASK),
               "S41_SERVER_FFN_MAX_TOKENS": "4", "S41_SERVER_FFN_N_EMBD": "5120",
               "S41_SERVER_FFN_RUNTIME_CONTROL": "1", "S41_SERVER_FFN_DORMANT_HOST_SHARE": "1",
               **self.usb.server_environment()}
        self.fake_gate = argparse.Namespace(
            phone_owner=lambda *args: (self.op15, "command", self.usb, env, OP15_MASK, ("HTP0-shard",)),
            run=lambda config, output, arm, options: None, main=lambda: None)
        self.config = {
            "gpu_layers": 16, "phone": {"serial": OP15_SERIAL, "session_masks": {"HTP0": 63, "HTP1": 4032,
                                                                                  "HTP2": 258048}},
            "helper_phone": {
                "device_id": "pixel10pro-phone", "label": "pixel", "serial": PIXEL_SERIAL, "adb_port": 5037,
                "worker_path": pixel_helper_json()["worker_path"],
                "library_directories": pixel_helper_json()["library_directories"],
                "shard_path": "/data/local/tmp/s42-pixel10pro-qualification-20260922-v1/HTP0.ffn.gguf",
                "layer_mask": PIXEL_MASK, "columns": 17408, "column_quantum": 4352, "max_tokens": 4,
                "backend": "Vulkan0", "phone_port": 26990, "max_requests": 100,
                "expected_sha256_by_path": {
                    pixel_helper_json()["worker_path"]: SHA,
                    "/data/local/tmp/s42-pixel10pro-qualification-20260922-v1/HTP0.ffn.gguf": SHA},
            },
        }

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _owner(self, config=None, *, with_helper=True):
        gate = self.module.install(self.fake_gate, with_helper=with_helper)
        with patch.object(self.module, "AdbTcpPhoneWorkerSession", FakeHelperSession):
            return gate.phone_owner(config or self.config, self.manifest, self.output,
                                    argparse.Namespace(plan_sha256=SHA), (750000,), True)

    def test_owner_composes_two_helpers_and_attributes_the_helper_calls(self) -> None:
        owner, _, _, env, mask, shards = self._owner()
        self.assertEqual(mask, OP15_MASK | PIXEL_MASK)
        self.assertEqual(shards[-1].session_id, "PIXEL0")
        self.assertEqual(shards[-1].layer_mask, PIXEL_MASK)
        self.assertEqual(env["S41_SERVER_FFN_LAYER_MASK"], str(OP15_MASK | PIXEL_MASK))
        owner.preflight()
        owner.start("command", self.manifest, self.usb)
        self.assertEqual(env["S41_SERVER_FFN_HELPERS"], "2")
        self.assertEqual(env["S41_SERVER_FFN_HELPER0_LAYER_MASK"], str(OP15_MASK))
        self.assertEqual(env["S41_SERVER_FFN_HELPER0_TRANSPORT"], "functionfs-usb")
        self.assertEqual(env["S41_SERVER_FFN_HELPER1_PORT"], "40317")
        self.assertNotIn("S41_SERVER_FFN_TRANSPORT", env)
        proofs = (argparse.Namespace(session_id="HTP0", to_json=lambda: {"session_id": "HTP0"}),
                  argparse.Namespace(session_id="PIXEL0", to_json=lambda: {"session_id": "PIXEL0"}))
        owner.record_execution_proof("request", SHA, proofs)
        self.assertEqual(self.op15.proofs, {"request": ["HTP0"]})
        second = 1 + (1 << 24)
        lines = [f"S41SERVERFFNCALL request={request} layer={layer} tokens=1 columns=13056 payload_bytes=10240"
                 for request, layer in ((1, 0), (2, 17), (second, 18), (second + 1, 23), (second + 2, 18))]
        (self.output / "SERVER_FFN_LINES.json").write_text(json.dumps(lines))
        close = owner.finish(require_execution=True).to_json()
        self.assertEqual(close["helper_served_calls"], 3)
        self.assertEqual(FakeHelperSession.instances[-1].stopped_with, 3)
        self.assertEqual(close["helper_proofs"], {"request": [{"session_id": "PIXEL0"}]})
        self.assertEqual(self.op15.events, ["preflight", "start", "finish"])

        (self.output / "RESULT.json").write_text(json.dumps({
            "status": "COMPLETED", "paid_s": 100.0, "paid_host_energy": {"joules": 1.0},
            "requests": [{"split": True, "decode_s": 40.0}]}))
        (self.output / "PHONE_CLOSE.json").write_text(json.dumps(close))
        summary = self.module.write_two_phone_result(self.config, self.output)
        self.assertEqual(summary["calls_by_device"]["pixel10pro-phone"]["calls"], 3)
        self.assertEqual(summary["calls_by_device"]["op15-phone"]["layers_seen"], [0, 17])
        self.assertAlmostEqual(summary["helper_assumed_j"]["assisting"], 40.0 * 4.5 + 60.0 * 0.875)

    def test_the_real_gate_exposes_the_wrapped_hooks(self) -> None:
        gate = self.module.load_gate()
        original_owner = gate.phone_owner
        self.module.install(gate, with_helper=True)
        self.assertIsNot(gate.phone_owner, original_owner)
        self.assertTrue(callable(gate.main) and callable(gate.split_phone_proofs))

    def test_op15_only_arm_and_invalid_helpers(self) -> None:
        owner, _, _, env, mask, _ = self._owner(with_helper=False)
        self.assertIs(owner, self.op15)
        self.assertEqual(mask, OP15_MASK)
        for change in ({"layer_mask": PIXEL_MASK | 1}, {"layer_mask": 1 << 30}, {"max_tokens": 8},
                       {"columns": 8704}):
            config = copy.deepcopy(self.config)
            config["helper_phone"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self._owner(config)


class PreflightRowsTests(unittest.TestCase):
    def test_helper_rows_add_usb_checks_and_a_blocked_dispatch_check(self) -> None:
        from research_dev.scheduler.campaigns.burstgpt import preflight
        args = argparse.Namespace(helper_phone=[json.dumps(pixel_helper_json())], phone_usb_serial=OP15_SERIAL,
                                  minimum_usb_speed_mbps=5000)
        rows = (type("Row", (), {"name": "phone-usb-port:pixel10pro-phone", "passed": True, "detail": "ok"})(),)
        with patch.object(preflight, "check_usb_topology", return_value=(rows, {})) as probe:
            checks = preflight._helper_phone_checks(args)
        self.assertEqual(probe.call_args.args[0][1], ("pixel10pro-phone", PIXEL_SERIAL, "2-9.2", 5000))
        self.assertEqual([row.to_json()["status"] for row in checks], ["PASS", "BLOCKED", "BLOCKED", "BLOCKED"])
        self.assertEqual([row.to_json()["check_id"] for row in checks][1:], [
            "helper-phone-transport-identity:pixel10pro-phone", "helper-phone-cost-evidence:pixel10pro-phone",
            "two-phone-dispatch"])
        self.assertEqual(preflight._helper_phone_checks(argparse.Namespace(helper_phone=[])), [])


if __name__ == "__main__":
    unittest.main()
