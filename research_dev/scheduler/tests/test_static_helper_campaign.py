"""Static co-helper lifecycle, evidence binding and separate energy domains."""

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from research_dev.scheduler import RuntimePhonePowerProfile
from research_dev.scheduler._internal.capability_contracts.common import _placement_profile_json
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler.adapters.co_helper_lifecycle import IdleCoHelperStopPolicy
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler.adapters.energy import PhoneActivityIntervalTracker, PolledPhonePowerSampler, RaplNvmlPhoneEnergyMeter
from research_dev.scheduler.adapters.heterogeneous_rig import HeterogeneousPhysicalRig
from research_dev.scheduler.adapters.http_backend import _AdaptivePayloadController
from research_dev.scheduler.adapters.phone_helpers import IDENTITY_REQUIREMENTS
from research_dev.scheduler.adapters.phone_transport import ADB_TCP_TRANSPORT_GENERATION
from research_dev.scheduler.adapters.preflight import _check_model_phone_state, _check_endpoint_contracts
from research_dev.scheduler.campaigns.burstgpt.helper_phone_evidence import (
    campaign_co_helper_lifecycles, digest, extend_helper_profile, extend_helper_overlay, load_helper_evidence,
)
import two_phone_harness as harness


class StaticHelperCampaignTests(unittest.TestCase):
    def bundle(self, directory):
        directory = Path(directory)
        host = directory / "server"
        host.write_bytes(b"server fixture")
        receipt = directory / "receipt.json"
        receipt.write_text('{"status":"PASS"}')
        raw = _placement_profile_json(harness.placement_profile())
        fragment = {key: [row for row in raw[key] if (
            (row == "energy:" + harness.PIXEL) if isinstance(row, str)
            else (row.get("device_id") == harness.PIXEL or row.get("pool_id") == "pixel-ram"
                  or row.get("domain_id") == "energy:" + harness.PIXEL
                  or harness.PIXEL in (row.get("source_device"), row.get("target_device"))))]
                    for key in ("devices", "memory_pools", "domains", "kernels", "links", "idle_charge_domains")}
        value = {
            "schema": "s42-static-helper-evidence-v1", "status": "PASS",
            "worker": {
                "device_id": harness.PIXEL, "serial": harness.PIXEL_SERIAL, "adb_port": 5037,
                "adb_path": "/usr/bin/adb", "worker_path": "/phone/worker", "library_directories": ["/phone"],
                "shard_path": "/phone/shard.ffn.gguf", "artifact_sha256": harness.EVIDENCE,
                "layer_mask": harness.PIXEL_MASK, "n_embd": 32, "columns": 128, "column_quantum": 32,
                "max_tokens": 4, "swiglu": True, "backend": "CPU", "phone_port": 26990,
                "forward_port": 26991, "max_requests": 0, "worker_environment": {},
                "expected_sha256_by_path": {"/phone/worker": harness.EVIDENCE,
                    "/phone/shard.ffn.gguf": harness.SHARD_SHA, "/phone/lib.so": harness.EVIDENCE},
            },
            "transport_identity": {
                "schema": "research-scheduler-phone-helper-transport-identity-v1", "device_id": harness.PIXEL,
                "transport": "adb-tcp", "transport_generation": ADB_TCP_TRANSPORT_GENERATION,
                "minimum_usb_speed_mbps": 5000,
                "hardware_identity": {"adb_usb_identity": "18d1:4ee7", "host_usb_controller": "controller",
                    "phone_kernel_release": "kernel", "phone_usb_serial": harness.PIXEL_SERIAL,
                    "phone_usb_sysfs_device": "2-9.2"},
                "software_identity": {"host_binary_sha256": digest(host), "host_impl_sha256": digest(host),
                    "transport_client_source_sha256": digest(host), "worker_environment_sha256": canonical_sha256({}),
                    "phone_worker_sha256": harness.EVIDENCE, "phone_shard_sha256": harness.SHARD_SHA,
                    "phone_library_sha256:/phone/lib.so": harness.EVIDENCE},
                "receipts": {kind: digest(receipt) for kind in IDENTITY_REQUIREMENTS["adb-tcp"]["receipts"]},
            },
            "receipt_paths": {kind: str(receipt) for kind in IDENTITY_REQUIREMENTS["adb-tcp"]["receipts"]},
            "host_software_paths": {key: str(host) for key in
                                    ("host_binary_sha256", "host_impl_sha256", "transport_client_source_sha256")},
            "profile_fragment": fragment,
            "power": RuntimePhonePowerProfile.assumed_4p5w(device_id=harness.PIXEL,
                domain_id="energy:" + harness.PIXEL, allow_assumed_for_scheduling=True).to_json(),
        }
        path = directory / "bundle.json"
        path.write_text(json.dumps(value))
        return path, value

    def test_pinned_evidence_adds_only_its_device_and_separate_assumed_power(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.bundle(directory)
            evidence = load_helper_evidence(path)
            base = harness.placement_profile(with_pixel=False)
            result = extend_helper_profile(base, {harness.PIXEL: evidence})
            self.assertIn(harness.PIXEL, result.devices)
            self.assertEqual(result.devices[harness.CPU], base.devices[harness.CPU])
            self.assertEqual(result.domains[evidence.power.domain_id],
                             harness.placement_profile().domains[evidence.power.domain_id])
            self.assertEqual(evidence.power.idle_power_mw, 875)

    def test_changed_host_binary_or_receipt_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.bundle(directory)
            (Path(directory) / "server").write_bytes(b"different build")
            with self.assertRaisesRegex(PhysicalAdapterError, "host software differs"):
                load_helper_evidence(path)
            path, _ = self.bundle(directory)
            (Path(directory) / "receipt.json").write_text('{"status":"FAIL"}')
            with self.assertRaisesRegex(PhysicalAdapterError, "receipt differs or failed"):
                load_helper_evidence(path)

    def test_existing_overlay_gains_only_the_qualified_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.bundle(directory)
            evidence = load_helper_evidence(path)
            model = harness.manifest(directory)
            base = harness.runtime_catalog(model, declaration=harness.co_helpers())
            overlay = harness.runtime_catalog(model)
            extended = extend_helper_overlay(base, overlay, {harness.PIXEL: evidence})
            self.assertEqual(extended.executor_by_device[harness.PIXEL], base.executor_by_device[harness.PIXEL])
            self.assertEqual(extended.executor_by_device[harness.OP15], overlay.executor_by_device[harness.OP15])
            self.assertEqual(extended.composite_executors, overlay.composite_executors)

    def test_catalog_without_helper_cannot_launch_an_extra_phone(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.bundle(directory)
            model = harness.manifest(directory)
            with self.assertRaisesRegex(PhysicalAdapterError, "catalog and helper"):
                campaign_co_helper_lifecycles([path], harness.runtime_catalog(model),
                    {model.model_id: model}, Path(directory) / "server")

    def test_preflight_checks_identify_each_phone_and_accept_only_helper_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            model = harness.manifest(directory)
            catalog = harness.runtime_catalog(model, declaration=harness.co_helpers())
            catalog = replace(catalog, executors=tuple(item if item.device_id == harness.PIXEL
                              else replace(item, endpoint=f"http://127.0.0.1:{19000 + index}")
                              for index, item in enumerate(catalog.executors)))
            row = SimpleNamespace(manifest=model, snapshot=harness.snapshot(model, catalog))
            checks = []
            _check_model_phone_state(checks, catalog, row)
            self.assertEqual(len(checks), 6)
            self.assertEqual(len({check.check_id for check in checks}), 6)
            _check_endpoint_contracts(checks, catalog)
            self.assertEqual(next(check.status for check in checks if check.check_id == "endpoint-contracts"), "PASS")
            invalid = replace(catalog, executors=tuple(replace(item, endpoint="physical://wrong-phone")
                              if item.device_id == harness.PIXEL else item for item in catalog.executors))
            checks = []
            _check_endpoint_contracts(checks, invalid)
            self.assertEqual(checks[0].status, "BLOCKED")

    def test_idle_stop_is_explicit(self):
        calls = []
        class Session:
            def stop(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(to_json=lambda: {"phase": "stop"})
        self.assertEqual(IdleCoHelperStopPolicy().stop(Session()), {"phase": "stop"})
        self.assertEqual(calls, [{"served_calls": None, "allow_idle_signal": True}])

    def test_startup_is_after_paid_epoch_and_receipts_survive_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            rig = HeterogeneousPhysicalRig.__new__(HeterogeneousPhysicalRig)
            rig.configuration = SimpleNamespace(output_directory=Path(directory))
            rig._lock = threading.RLock()
            rig._transition_active = False
            rig._co_helper_receipts = []
            def start(path):
                self.assertEqual(rig.epoch_ns, 123)
                self.assertTrue(path.is_dir())
                return ({"phase": "launch"},)
            rig._co_helper_lifecycles = {harness.EVIDENCE: SimpleNamespace(
                start_trace=start, end_trace=lambda _: ({"phase": "stop"},))}
            rig.begin_trace(123)
            rig._stop_co_helpers()
            self.assertEqual(json.loads((Path(directory) / "CO_HELPER_LIFECYCLE.json").read_text()),
                             [{"phase": "launch"}, {"phase": "stop"}])

    def test_helper_energy_uses_union_without_double_charging_host(self):
        rows = [{key: {"sample_t_ns": ns} for key in ("gpu", "rapl_package")}
                for ns in (0, 4_000_000_000)]
        helper = PhoneActivityIntervalTracker()
        meter = RaplNvmlPhoneEnergyMeter(lambda: rows,
            lambda *_: {"cpu_package_energy_j": 1, "gpu_board_energy_j": 2},
            PolledPhonePowerSampler(lambda: None), energy_boundary_id="test-fleet",
            phone_power_profile=RuntimePhonePowerProfile.assumed_4p5w(device_id="op15", domain_id="op15-system"),
            phone_activity=PhoneActivityIntervalTracker(),
            helper_phone_power={"pixel": (RuntimePhonePowerProfile.assumed_4p5w(
                device_id="pixel", domain_id="pixel-system"), helper)})
        meter.record_helper_phone_window("a", 1_000_000_000, 2_000_000_000, ("pixel",))
        meter.record_helper_phone_window("b", 1_500_000_000, 2_500_000_000, ("pixel",))
        result = meter.measure(1_000_000_000, 3_000_000_000)
        self.assertEqual(dict(result.fleet_energy_uj_by_domain), {
            "cpu-package": 1_000_000, "gpu-board": 2_000_000,
            "op15-system": 1_750_000, "pixel-system": 7_187_500})
        self.assertEqual(result.estimation_metadata["helper:pixel:active_time_ns"], 1_500_000_000)

    def test_single_phone_windows_do_not_require_co_helper_bindings(self):
        primary, helper = [], []
        controller = _AdaptivePayloadController.__new__(_AdaptivePayloadController)
        controller.backend = SimpleNamespace(_epoch_ns=1000, _energy_meter=SimpleNamespace(
            record_phone_activity_duration=lambda *row: primary.append(row),
            record_helper_phone_window=lambda *row: helper.append(row)))
        controller.command = SimpleNamespace(adapter_parameters={})
        delta = {"rpc_us": 5, "usb_transfer_us": 1, "phone_compute_us": 4}
        controller._record_phone_window("gemma", 10, 20, delta)
        self.assertEqual(primary[0][-1], 5)
        self.assertEqual(helper[0][-1], ())
        controller._record_phone_window("qwen", 20, 30, delta,
            SimpleNamespace(device_layer_masks=((harness.OP15, 63), (harness.PIXEL, harness.PIXEL_MASK))))
        self.assertEqual(helper[1][-1], (harness.OP15, harness.PIXEL))


if __name__ == "__main__":
    unittest.main()
