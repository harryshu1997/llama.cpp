#!/usr/bin/env python3

from __future__ import annotations

import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SUBJECT = HERE.parent / "capture_a_only_inventory_v1.py"
MATERIALIZER = HERE.parent / "materialize_a_only_inputs_v1.py"
USB_LAUNCHER = HERE.parent / "managed_runtime_launcher_usb_v1.py"
MANAGED_LAUNCHER = (
    HERE.parents[2]
    / "v23_readiness"
    / "a_only_acquisition_driver_v1"
    / "producers_v1"
    / "managed_runtime_launcher_v1.py"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = load(SUBJECT, "capture_a_only_inventory_v1_test")
materializer = load(MATERIALIZER, "capture_a_only_materializer_test")
managed_launcher = load(MANAGED_LAUNCHER, "capture_a_only_managed_launcher_test")
usb_launcher = load(USB_LAUNCHER, "capture_a_only_usb_launcher_test")


def android_stat(
    *,
    inode: int,
    size: int,
    mode: int = stat.S_IFREG | 0o755,
    second: int = 1_700_000_000,
) -> bytes:
    timestamp = datetime.datetime.fromtimestamp(
        second,
        datetime.timezone.utc,
    ).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"DEV=7|INO={inode}|SIZE={size}|MODE={mode:x}|"
        f"MTIME_S={second}|MTIME={timestamp}.123456789 +0000|"
        f"CTIME_S={second}|CTIME={timestamp}.987654321 +0000\n"
    ).encode("ascii")


class FakeRunner:
    def __init__(self):
        self.identities = {
            endpoint: {
                key: value
                for key, value in phone.items()
                if key in {"serial", "product", "model", "device"}
            }
            for endpoint, phone in capture.PHONES.items()
        }
        self.files = {}
        self.names = {}
        self.mutate_stat = set()
        self.mutate_digest = set()
        self.digest_calls = {}
        inode = 100
        for endpoint, bundle_id, root, files in (
            ("op12", "op12_stagenet", capture.OP12_STAGE_ROOT, self.stage_files("op12")),
            ("op15", "op15_stagenet", capture.OP15_STAGE_ROOT, self.stage_files("op15")),
            ("op15", "op15_direct_relay", capture.OP15_RELAY_ROOT, capture.RELAY_FILES),
        ):
            del bundle_id
            self.names[(endpoint, root)] = sorted(files)
            for name in sorted(files):
                inode += 1
                path = f"{root}/{name}"
                raw = f"{endpoint}:{path}\n".encode("ascii")
                self.files[(endpoint, path)] = {
                    "digest": hashlib.sha256(raw).hexdigest(),
                    "inode": inode,
                    "mode": stat.S_IFREG | 0o755,
                    "size": len(raw),
                }

    @staticmethod
    def stage_files(endpoint):
        result = dict(capture.PHONE_STAGE_FILES)
        bundle_id = f"{endpoint}_stagenet"
        name, component = capture.PHONE_STAGE_HVX[bundle_id]
        result[name] = (component, "backend_library")
        return result

    def run(self, argv, timeout):
        del timeout
        self.assert_prefix(argv)
        serial = argv[argv.index("-s") + 1]
        endpoint = next(
            name
            for name, value in capture.PHONES.items()
            if value["serial"] == serial
        )
        tail = argv[argv.index(serial) + 1 :]
        if tail[:2] == ["shell", "getprop"]:
            property_name = tail[2]
            field = {
                "ro.serialno": "serial",
                "ro.product.name": "product",
                "ro.product.model": "model",
                "ro.product.device": "device",
            }[property_name]
            return (self.identities[endpoint][field] + "\n").encode("ascii")
        command = tail[1] if tail[:1] == ["shell"] else ""
        if "ls -1A" in command:
            root = next(
                root
                for candidate_endpoint, root in self.names
                if candidate_endpoint == endpoint and root in command
            )
            return ("\n".join(self.names[(endpoint, root)]) + "\n").encode("ascii")
        path = next(
            path
            for candidate_endpoint, path in self.files
            if candidate_endpoint == endpoint and path in command
        )
        row = self.files[(endpoint, path)]
        if "stat -c" in command:
            inode = row["inode"] + (
                1 if (endpoint, path) in self.mutate_stat else 0
            )
            self.mutate_stat.discard((endpoint, path))
            return android_stat(
                inode=inode,
                size=row["size"],
                mode=row["mode"],
            )
        if "sha256sum" in command:
            key = (endpoint, path)
            self.digest_calls[key] = self.digest_calls.get(key, 0) + 1
            digest = (
                "0" * 64
                if key in self.mutate_digest and self.digest_calls[key] == 2
                else row["digest"]
            )
            return f"{digest}  {path}\n".encode("ascii")
        raise AssertionError(argv)

    @staticmethod
    def assert_prefix(argv):
        if argv[:3] != [capture.ADB_PATH, "-P", "5038"] or "-s" not in argv:
            raise AssertionError(argv)


class Fixture:
    def __init__(self, case):
        temporary = tempfile.TemporaryDirectory()
        case.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.monolithic_root = self.root / "mono"
        self.cuda_root = self.root / "route"
        self.monolithic_root.mkdir()
        self.cuda_root.mkdir()
        self.original_roots = copy.deepcopy(capture.ROOTS)
        self.original_mono = capture.CUDA_MONOLITHIC_ROOT
        self.original_cuda = capture.CUDA_ROUTE_ROOT
        capture.CUDA_MONOLITHIC_ROOT = str(self.monolithic_root)
        capture.CUDA_ROUTE_ROOT = str(self.cuda_root)
        capture.ROOTS["cuda_monolithic"] = str(self.monolithic_root)
        capture.ROOTS["cuda_route"] = str(self.cuda_root)
        case.addCleanup(self.restore)
        for name in capture.CUDA_ROUTE_FILES:
            path = self.cuda_root / name
            path.write_bytes((name + "\n").encode("ascii"))
            path.chmod(0o755)
        required = []
        for index, name in enumerate(("llama-layersplit", "libggml.so"), 1):
            path = self.monolithic_root / name
            path.write_bytes((name + "\n").encode("ascii"))
            path.chmod(0o755)
            pin = capture.secure_local_pin(
                path,
                executable=name == "llama-layersplit",
            )
            required.append(
                {
                    "component_id": "cuda-mono.bin" if index == 1 else "cuda-mono.lib",
                    "path": str(path),
                    "sha256": pin["sha256"],
                    "stat": pin["stat"],
                }
            )
        self.launch = self.root / "cuda-monolithic-launch.json"
        self.launch.write_bytes(
            capture.canonical_bytes(
                {
                    "bundle_root": str(self.monolithic_root),
                    "launcher_component_id": "cuda-mono.bin",
                    "model_sha256": capture.MODEL_SHA256,
                    "required_components": required,
                    "schema": "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
                }
            )
        )
        self.runner = FakeRunner()

    def restore(self):
        capture.CUDA_MONOLITHIC_ROOT = self.original_mono
        capture.CUDA_ROUTE_ROOT = self.original_cuda
        capture.ROOTS.clear()
        capture.ROOTS.update(self.original_roots)

    def inventory(self):
        return capture.capture_runtime_inventory(
            self.launch,
            runner=self.runner,
        )


class InventoryTests(unittest.TestCase):
    def test_exact_closure_and_codec(self):
        fixture = Fixture(self)
        value = fixture.inventory()
        self.assertEqual(value["schema"], capture.RUNTIME_SCHEMA)
        self.assertTrue(value["closure_complete"])
        codec = next(
            item for item in value["components"]
            if item["component_id"] == "cuda-tokenize"
        )
        self.assertTrue(codec["path"].endswith("/llama-token-codec"))
        relay = next(
            item for item in value["bundles"]
            if item["bundle_id"] == "op15_direct_relay"
        )
        self.assertEqual(len(relay["required_component_ids"]), 1)

    def test_wrong_serial_is_rejected(self):
        fixture = Fixture(self)
        fixture.runner.identities["op12"]["serial"] = "wrong"
        with self.assertRaisesRegex(capture.CaptureError, "phone.op12.serial"):
            fixture.inventory()

    def test_missing_required_file_is_rejected(self):
        fixture = Fixture(self)
        fixture.runner.names[("op12", capture.OP12_STAGE_ROOT)].pop()
        with self.assertRaisesRegex(capture.CaptureError, "closure.op12_stagenet"):
            fixture.inventory()

    def test_extra_required_file_is_rejected(self):
        fixture = Fixture(self)
        fixture.runner.names[("op15", capture.OP15_STAGE_ROOT)].append("extra.so")
        fixture.runner.names[("op15", capture.OP15_STAGE_ROOT)].sort()
        with self.assertRaisesRegex(capture.CaptureError, "closure.op15_stagenet"):
            fixture.inventory()

    def test_remote_stat_mutation_is_rejected(self):
        fixture = Fixture(self)
        path = f"{capture.OP12_STAGE_ROOT}/llama-layersplit"
        fixture.runner.mutate_stat.add(("op12", path))
        with self.assertRaisesRegex(capture.CaptureError, "remote.mutation"):
            fixture.inventory()

    def test_remote_digest_mutation_is_rejected(self):
        fixture = Fixture(self)
        path = f"{capture.OP12_STAGE_ROOT}/llama-layersplit"
        fixture.runner.mutate_digest.add(("op12", path))
        with self.assertRaisesRegex(capture.CaptureError, "remote.digest_mutation"):
            fixture.inventory()

    def test_local_digest_mutation_is_rejected(self):
        fixture = Fixture(self)
        launch = json.loads(fixture.launch.read_text(encoding="ascii"))
        launch["required_components"][0]["sha256"] = "0" * 64
        fixture.launch.write_bytes(capture.canonical_bytes(launch))
        with self.assertRaisesRegex(capture.CaptureError, "monolithic.cuda-mono.bin"):
            fixture.inventory()

    def test_local_symlink_is_rejected(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        target = root / "target"
        target.write_bytes(b"x")
        alias = root / "alias"
        alias.symlink_to(target)
        with self.assertRaisesRegex(capture.CaptureError, "E_LOCAL_NOT_REGULAR"):
            capture.secure_local_pin(alias)

    def test_symlinked_local_bundle_root_is_rejected(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        target = root / "target"
        target.mkdir()
        (target / "worker").write_bytes(b"x")
        alias = root / "alias"
        alias.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(
            capture.CaptureError,
            "E_LOCAL_DIRECTORY_TYPE",
        ):
            capture._local_components(
                alias,
                "cuda_route",
                "cuda",
                {"worker": ("cuda-route.bin", "executable")},
            )

    def test_extra_monolithic_file_is_rejected(self):
        fixture = Fixture(self)
        launch = json.loads(fixture.launch.read_text(encoding="ascii"))
        root = Path(launch["bundle_root"])
        (root / "unexpected.so").write_bytes(b"extra")
        with self.assertRaisesRegex(
            capture.CaptureError,
            "closure.cuda_monolithic",
        ):
            fixture.inventory()

    def test_remote_nonregular_is_rejected(self):
        fixture = Fixture(self)
        path = f"{capture.OP15_STAGE_ROOT}/llama-layersplit"
        fixture.runner.files[("op15", path)]["mode"] = stat.S_IFDIR | 0o755
        with self.assertRaisesRegex(capture.CaptureError, "E_REMOTE_NOT_REGULAR"):
            fixture.inventory()

    def test_relay_fake_library_is_rejected(self):
        fixture = Fixture(self)
        fixture.runner.names[("op15", capture.OP15_RELAY_ROOT)].append("libfake.so")
        fixture.runner.names[("op15", capture.OP15_RELAY_ROOT)].sort()
        with self.assertRaisesRegex(capture.CaptureError, "closure.op15_direct_relay"):
            fixture.inventory()

    def test_noncanonical_input_is_rejected(self):
        fixture = Fixture(self)
        value = json.loads(fixture.launch.read_text(encoding="ascii"))
        fixture.launch.write_text(json.dumps(value, indent=2), encoding="ascii")
        with self.assertRaisesRegex(capture.CaptureError, "canonical"):
            fixture.inventory()


class AuthorityTests(unittest.TestCase):
    def test_correct_local_plan_is_two_level(self):
        fixture = Fixture(self)
        runtime = fixture.inventory()
        launcher_path = fixture.root / "managed-launcher"
        launcher_path.write_bytes(b"#!/bin/sh\n")
        launcher_path.chmod(0o755)
        launcher = capture.secure_local_pin(launcher_path, executable=True)
        plan, argv = capture.build_local_cuda_plan(runtime, launcher)
        self.assertEqual(argv[0], str(launcher_path))
        self.assertEqual(
            argv[1::2],
            ["--plan-json", "--plan-sha256", "--boot-id"],
        )
        self.assertNotIn(runtime["bundle_roots"]["cuda_route"] + "/llama-layersplit", argv)
        self.assertEqual(plan["route"]["kind"], "local_exec")
        self.assertEqual(plan["route"]["argv"][0], runtime["bundle_roots"]["cuda_route"] + "/llama-layersplit")
        self.assertIsNone(plan["android"])
        self.assertIsNone(plan["ssh"])
        self.assertEqual(plan["mode"], "local_cuda")
        managed_launcher.validate_plan(copy.deepcopy(plan))

    def test_current_authority_refuses_executable_shapes(self):
        fixture = Fixture(self)
        runtime = fixture.inventory()
        launcher_path = fixture.root / "managed-launcher"
        launcher_path.write_bytes(b"#!/bin/sh\n")
        launcher_path.chmod(0o755)
        usb_path = fixture.root / "usb-launcher"
        usb_path.write_bytes(b"#!/bin/sh\n")
        usb_path.chmod(0o755)
        launcher = capture.secure_local_pin(launcher_path, executable=True)
        usb = capture.secure_local_pin(usb_path, executable=True)
        _plan, argv = capture.build_local_cuda_plan(runtime, launcher)
        processes = capture.build_phone_processes(
            runtime,
            usb,
            {
                "path": capture.ADB_PATH,
                "sha256": "a" * 64,
            },
        )
        phone_inline = processes["op12_stagenet"]["argv"]
        phone_raw = phone_inline[phone_inline.index("--plan-json") + 1]
        phone_sha = phone_inline[phone_inline.index("--plan-sha256") + 1]
        usb_launcher.validate_plan(
            phone_raw,
            phone_sha,
            managed_launcher,
        )
        phone = {
            "mechanism_commands": {
                "desktop": [["/x"] for _ in range(9)],
                "op12": [],
                "op15": [],
            },
            "processes": processes,
        }
        cuda = {
            "codec": {
                "argv": ["/codec", "--model", capture.MODEL_PATH, "--model-sha256", capture.MODEL_SHA256],
                "cwd": str(fixture.root),
                "environment": {},
                "executable": {"bytes": 1, "path": "/codec", "sha256": "b" * 64, "stat": {"ctime_ns": 1, "device_id": 1, "inode": 1, "mode": stat.S_IFREG | 0o755, "mtime_ns": 1, "size": 1}},
                "timeout_ms": 1,
            },
            "expected_capabilities": 0x3F,
            "expected_file_type": 15,
            "expected_max_streams": 8,
            "expected_n_batch": 64,
            "expected_n_ctx_seq": 512,
            "expected_n_embd": 5120,
            "expected_n_layer": 40,
            "expected_n_ubatch": 64,
            "host": "127.0.0.1",
            "io_timeout_ms": 1,
            "mechanism_commands": phone["mechanism_commands"],
            "model_artifact": {"bytes": capture.MODEL_BYTES, "path": capture.MODEL_PATH, "sha256": capture.MODEL_SHA256, "stat": {"ctime_ns": 1, "device_id": 1, "inode": 2, "mode": stat.S_IFREG | 0o644, "mtime_ns": 1, "size": capture.MODEL_BYTES}},
            "nvidia_smi": {
                "device_argv": ["/nvidia"],
                "executable": {"bytes": 1, "path": "/nvidia", "sha256": "c" * 64, "stat": {"ctime_ns": 1, "device_id": 1, "inode": 3, "mode": stat.S_IFREG | 0o755, "mtime_ns": 1, "size": 1}},
                "process_argv": ["/nvidia"],
                "timeout_ms": 1,
            },
            "port": capture.PORTS["cuda_route"],
            "route_epoch": 1,
            "worker": {
                "argv": argv,
                "cwd": str(fixture.cuda_root),
                "environment": {
                    "LAYERSPLIT_MEMORY_CERT": "1",
                    "LAYERSPLIT_MODEL_SHA256": capture.MODEL_SHA256,
                    "LAYERSPLIT_PLACEMENT_CERT": "1",
                },
                "executable": launcher,
                "runtime_component_ids": capture._bundle(runtime, "cuda_route")["required_component_ids"],
                "runtime_executable": {
                    key: value
                    for key, value in next(
                        item
                        for item in runtime["components"]
                        if item["component_id"] == "cuda-route.bin"
                    ).items()
                    if key in {"bytes", "path", "sha256", "stat"}
                },
                "shutdown_timeout_ms": 1,
                "startup_timeout_ms": 1,
            },
        }
        blockers = capture.current_authority_blockers(
            materializer=materializer,
            cuda_static=cuda,
            phone_static=phone,
        )
        self.assertEqual(
            blockers,
            [
                "E_V24_FLATTENED_CUDA_ARGV",
                "E_V24_PHONE_PLAN_OMITS_SSH",
            ],
        )


if __name__ == "__main__":
    unittest.main()
