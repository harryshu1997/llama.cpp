#!/usr/bin/env python3

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "managed_runtime_launcher_usb_v1.py"
)
wrapper = types.ModuleType("managed_runtime_launcher_usb_v1")
wrapper.__file__ = str(SOURCE)
exec(compile(SOURCE.read_bytes(), str(SOURCE), "exec"), wrapper.__dict__)


BOOT_ID = "11111111-1111-4111-8111-111111111111"
ADB = "/usr/bin/adb"
STAMP_NS = 1_704_067_200_000_000_000


def stat_row(size=10, include_build_id=True):
    value = {
        "ctime_ns": STAMP_NS,
        "device_id": 1,
        "inode": 100 + size,
        "mode": stat.S_IFREG | 0o755,
        "mtime_ns": STAMP_NS,
        "size": size,
    }
    if include_build_id:
        value["build_id"] = None
    return value


def worker_plan(endpoint="op15", include_build_id=True):
    serial = wrapper.ENDPOINT_SERIALS[endpoint]
    runtime_root = f"/data/local/tmp/s39-v24/{endpoint}"
    launcher_path = f"{runtime_root}/llama-layersplit"
    return {
        "android": {
            "adb_path": ADB,
            "adb_port": wrapper.ADB_PORT,
            "adb_selector": serial,
            "adb_sha256": "a" * 64,
            "boot_id_source": "phase_fresh_snapshot",
            "physical_serial": serial,
            "shutdown_timeout_ms": 1000,
            "startup_timeout_ms": 30000,
        },
        "bundle_id": f"{endpoint}_stagenet",
        "components": [
            {
                "bytes": 10,
                "component_id": f"{endpoint}.launcher",
                "path": launcher_path,
                "sha256": "b" * 64,
                "stat": stat_row(include_build_id=include_build_id),
            },
        ],
        "endpoint": endpoint,
        "launcher_component_id": f"{endpoint}.launcher",
        "mode": "android",
        "route": {
            "devices": "HTP0",
            "driver_batch": 64,
            "driver_context": 512,
            "driver_max_prefill": 32,
            "dynamic_cut": True,
            "kind": "stagenet_worker",
            "kv_unified": True,
            "layer_end": 30 if endpoint == "op15" else 24,
            "layer_start": 0,
            "mode": "stagenet",
            "model_path": f"/data/local/tmp/s39-v24/{endpoint}.gguf",
            "model_sha256": "c" * 64,
            "n_gpu_layers": 99,
            "placement_cert": True,
            "port": 40000 if endpoint == "op15" else 40001,
            "runtime_root": runtime_root,
        },
        "schema": "s39-managed-runtime-launch-plan-v1",
        "ssh": None,
    }


def encode_plan(value):
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return raw, hashlib.sha256(raw.encode("ascii")).hexdigest()


class ManagedRuntimeLauncherUsbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = wrapper.load_frozen_launcher()

    def validate(self, value):
        raw, checksum = encode_plan(value)
        launcher = wrapper.load_frozen_launcher()
        return wrapper.validate_plan(raw, checksum, launcher), launcher

    def assert_refused(self, value, needle=None):
        raw, checksum = encode_plan(value)
        launcher = wrapper.load_frozen_launcher()
        with self.assertRaises(wrapper.UsbLauncherError) as caught:
            wrapper.validate_plan(raw, checksum, launcher)
        if needle is not None:
            self.assertIn(needle, str(caught.exception))

    def test_frozen_source_is_exactly_pinned(self):
        raw = wrapper.read_frozen_source()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            wrapper.FROZEN_SOURCE_SHA256,
        )
        self.assertEqual(
            wrapper.FROZEN_SOURCE_SHA256,
            "b97941dc30399135b04e98dbdf102aaeb6c695c55a7b990402aeafd21ed4a245",
        )

    def test_modified_frozen_source_is_rejected(self):
        raw = wrapper.FROZEN_SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launcher.py"
            path.write_bytes(raw + b"\n")
            with self.assertRaisesRegex(
                wrapper.UsbLauncherError,
                "frozen_source.sha256",
            ):
                wrapper.read_frozen_source(
                    path,
                    wrapper.FROZEN_SOURCE_SHA256,
                )

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW unavailable")
    def test_symlinked_frozen_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launcher.py"
            path.symlink_to(wrapper.FROZEN_SOURCE)
            with self.assertRaisesRegex(
                wrapper.UsbLauncherError,
                "E_FROZEN_SOURCE_OPEN",
            ):
                wrapper.read_frozen_source(
                    path,
                    wrapper.FROZEN_SOURCE_SHA256,
                )

    def test_both_exact_usb_identities_are_accepted(self):
        for endpoint, serial in wrapper.ENDPOINT_SERIALS.items():
            with self.subTest(endpoint=endpoint):
                plan, launcher = self.validate(worker_plan(endpoint))
                self.assertEqual(plan["android"]["adb_selector"], serial)
                self.assertEqual(plan["android"]["physical_serial"], serial)
                self.assertEqual(
                    launcher.adb_prefix(plan["android"]),
                    [ADB, "-P", "5038", "-s", serial],
                )

    def test_missing_component_build_id_is_normalized_only_to_none(self):
        value = worker_plan(include_build_id=False)
        plan, _launcher = self.validate(value)
        self.assertIsNone(plan["components"][0]["stat"]["build_id"])
        self.assertNotIn("build_id", value["components"][0]["stat"])

    def test_non_null_build_id_is_rejected_by_frozen_validator(self):
        value = worker_plan()
        value["components"][0]["stat"]["build_id"] = "forged"
        self.assert_refused(value, "build_id")

    def test_other_missing_stat_field_is_not_normalized(self):
        value = worker_plan(include_build_id=False)
        del value["components"][0]["stat"]["inode"]
        self.assert_refused(value, "E_KEYS")

    def test_wifi_selector_is_rejected(self):
        value = worker_plan()
        value["android"]["adb_selector"] = "172.20.173.218:5555"
        self.assert_refused(value, "android.adb_selector")

    def test_default_adb_server_is_rejected(self):
        value = worker_plan()
        value["android"]["adb_port"] = 5037
        self.assert_refused(value, "android.adb_port")

    def test_cross_device_serial_is_rejected(self):
        value = worker_plan("op12")
        value["android"]["physical_serial"] = wrapper.ENDPOINT_SERIALS["op15"]
        value["android"]["adb_selector"] = wrapper.ENDPOINT_SERIALS["op15"]
        self.assert_refused(value, "android.physical_serial")

    def test_selector_must_equal_physical_serial(self):
        value = worker_plan("op15")
        value["android"]["adb_selector"] = wrapper.ENDPOINT_SERIALS["op12"]
        self.assert_refused(value, "android.adb_selector")

    def test_unknown_endpoint_is_rejected(self):
        value = worker_plan()
        value["endpoint"] = "op99"
        self.assert_refused(value, "E_ENDPOINT")

    def test_non_android_mode_is_rejected_before_delegate(self):
        value = worker_plan()
        value["mode"] = "local_cuda"
        self.assert_refused(value, "plan.mode")

    def test_remaining_route_validation_is_delegated(self):
        value = worker_plan()
        value["route"]["driver_batch"] = 0
        self.assert_refused(value, "route.driver_batch")

    def test_plan_digest_mutation_is_rejected(self):
        raw, checksum = encode_plan(worker_plan())
        mutated = raw.replace('"driver_batch":64', '"driver_batch":63')
        with self.assertRaisesRegex(
            wrapper.UsbLauncherError,
            "plan.sha256",
        ):
            wrapper.validate_plan(mutated, checksum, wrapper.load_frozen_launcher())

    def test_noncanonical_plan_is_rejected(self):
        raw = json.dumps(worker_plan(), indent=2)
        checksum = hashlib.sha256(raw.encode("ascii")).hexdigest()
        with self.assertRaisesRegex(
            wrapper.UsbLauncherError,
            "plan.canonical",
        ):
            wrapper.validate_plan(raw, checksum, wrapper.load_frozen_launcher())

    def test_duplicate_plan_key_is_rejected(self):
        raw = '{"endpoint":"op15","endpoint":"op12"}'
        checksum = hashlib.sha256(raw.encode("ascii")).hexdigest()
        with self.assertRaisesRegex(
            wrapper.UsbLauncherError,
            "E_DUPLICATE_KEY",
        ):
            wrapper.validate_plan(raw, checksum, wrapper.load_frozen_launcher())

    def test_execute_delegates_to_frozen_android_launcher(self):
        value = worker_plan("op12")
        raw, checksum = encode_plan(value)
        launcher = wrapper.load_frozen_launcher()
        runner = object()
        with mock.patch.object(
            launcher,
            "launch_android",
            return_value=17,
        ) as delegated:
            result = wrapper.execute(
                raw,
                checksum,
                BOOT_ID,
                launcher=launcher,
                runner=runner,
            )
        self.assertEqual(result, 17)
        delegated.assert_called_once()
        plan, actual_runner, boot_id = delegated.call_args.args
        self.assertIs(actual_runner, runner)
        self.assertEqual(boot_id, BOOT_ID)
        self.assertEqual(
            launcher.adb_prefix(plan["android"]),
            [ADB, "-P", "5038", "-s", "5ae7a43d"],
        )

    def test_cli_refuses_without_traceback(self):
        value = worker_plan()
        value["android"]["adb_port"] = 5037
        raw, checksum = encode_plan(value)
        stderr = io.StringIO()
        with mock.patch.object(sys, "stderr", stderr):
            result = wrapper.main(
                [
                    "--plan-json",
                    raw,
                    "--plan-sha256",
                    checksum,
                    "--boot-id",
                    BOOT_ID,
                ]
            )
        self.assertEqual(result, 2)
        self.assertIn("MANAGED_RUNTIME_USB_LAUNCH_REFUSED", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_unexpected_delegate_error_is_fail_closed_without_traceback(self):
        value = worker_plan()
        raw, checksum = encode_plan(value)
        launcher = wrapper.load_frozen_launcher()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                wrapper,
                "load_frozen_launcher",
                return_value=launcher,
            ),
            mock.patch.object(
                launcher,
                "launch_android",
                side_effect=RuntimeError("sensitive detail"),
            ),
            mock.patch.object(sys, "stderr", stderr),
        ):
            result = wrapper.main(
                [
                    "--plan-json",
                    raw,
                    "--plan-sha256",
                    checksum,
                    "--boot-id",
                    BOOT_ID,
                ]
            )
        self.assertEqual(result, 2)
        self.assertIn("E_UNEXPECTED_RuntimeError", stderr.getvalue())
        self.assertNotIn("sensitive detail", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
