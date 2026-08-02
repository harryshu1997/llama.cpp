#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SUBJECT = HERE.parent / "materialize_a_only_inputs_v1.py"
V24 = HERE.parents[1]
S39 = V24.parent


def load():
    spec = importlib.util.spec_from_file_location("test_a_only_adapter", SUBJECT)
    if spec is None or spec.loader is None:
        raise RuntimeError(SUBJECT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load()


def load_topology_helpers():
    source = HERE / "test_verify_topology_v1.py"
    spec = importlib.util.spec_from_file_location("test_adapter_topology_helpers", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


topology_helpers = load_topology_helpers()


def canonical(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class Fixture:
    def __init__(self, case: unittest.TestCase):
        temporary = tempfile.TemporaryDirectory()
        case.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = {
            "candidate": S39 / "CP0_R1_CANDIDATE.json",
            "contract": V24 / "CP0_R1_EVIDENCE_CONTRACT_V2_4.json",
            "cuda_monolithic_launch": (
                V24
                / "results"
                / "prephase_20260726T0915Z"
                / "cuda-monolithic-launch.json"
            ),
            "token_history": (
                V24
                / "results"
                / "prephase_20260726T0915Z"
                / "token-history.json"
            ),
            "tokenizer_plan": (
                V24
                / "results"
                / "prephase_20260726T0915Z"
                / "tokenizer-plan.json"
            ),
        }
        self.runtime_inventory = self._runtime_inventory()
        self.inputs["runtime_bundle_inventory"] = self.write(
            "runtime-inventory.json",
            self.runtime_inventory,
        )
        contract = json.loads(
            self.inputs["contract"].read_text(encoding="ascii")
        )
        self.topology = topology_helpers.verify.capture_topology(
            contract,
            runner=topology_helpers.FakeRunner(
                topology_helpers.runner_rows()
            ),
            clock_ns=topology_helpers.Counter(),
        )
        self.inputs["topology_receipt"] = self.write("topology.json", self.topology)
        self.operator = self._operator()
        self.inputs["operator_input"] = self.write("operator.json", self.operator)
        self.inventory = self._inventory()
        self.inventory_path = self.write("inventory.json", self.inventory)

    def write(self, name: str, value) -> Path:
        path = self.root / name
        path.write_bytes(canonical(value))
        return path

    @staticmethod
    def pin(path: Path):
        raw = path.read_bytes()
        return {
            "bytes": len(raw),
            "path": str(path),
            "sha256": sha(raw),
        }

    @staticmethod
    def pin_with_stat(path: Path):
        value = Fixture.pin(path)
        metadata = path.stat(follow_symlinks=False)
        value["stat"] = {
            "ctime_ns": metadata.st_ctime_ns,
            "device_id": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": metadata.st_mode,
            "mtime_ns": metadata.st_mtime_ns,
            "size": metadata.st_size,
        }
        return value

    @staticmethod
    def artifact(path: str, value: bytes = b"x\n"):
        return {
            "bytes": len(value),
            "path": path,
            "sha256": sha(value),
            "stat": {
                "ctime_ns": 1,
                "device_id": 1,
                "inode": len(path) + 100,
                "mode": stat.S_IFREG | 0o755,
                "mtime_ns": 1,
                "size": len(value),
            },
        }

    @staticmethod
    def component(bundle_id, endpoint, component_id, path, role):
        raw = f"{component_id}\n".encode("ascii")
        return {
            "bundle_id": bundle_id,
            "bytes": len(raw),
            "component_id": component_id,
            "endpoint": endpoint,
            "path": path,
            "role": role,
            "sha256": sha(raw),
            "stat": {
                "ctime_ns": 1,
                "device_id": 1,
                "inode": len(component_id) + 100,
                "mode": stat.S_IFREG | (0o755 if role == "executable" else 0o644),
                "mtime_ns": 1,
                "size": len(raw),
            },
        }

    def _runtime_inventory(self):
        monolithic = json.loads(
            self.inputs["cuda_monolithic_launch"].read_text(encoding="ascii")
        )
        cuda_route_root = self.root / "runtime-cuda-route"
        cuda_route_root.mkdir()
        roots = {
            "cuda_monolithic": monolithic["bundle_root"],
            "cuda_route": str(cuda_route_root),
            "op12_stagenet": "/data/local/tmp/s39/op12",
            "op15_direct_relay": "/data/local/tmp/s39/op15-relay",
            "op15_stagenet": "/data/local/tmp/s39/op15-stage",
        }
        endpoints = {
            "cuda_monolithic": "cuda",
            "cuda_route": "cuda",
            "op12_stagenet": "op12",
            "op15_direct_relay": "op15",
            "op15_stagenet": "op15",
        }
        components = []
        bundles = []
        for bundle_id in sorted(adapter.RUNTIME_BUNDLE_IDS):
            endpoint = endpoints[bundle_id]
            if bundle_id == "cuda_monolithic":
                launcher = monolithic["launcher_component_id"]
                required_ids = []
                for item in monolithic["required_components"]:
                    component_id = item["component_id"]
                    required_ids.append(component_id)
                    components.append(
                        {
                            "bundle_id": bundle_id,
                            "bytes": item["stat"]["size"],
                            "component_id": component_id,
                            "endpoint": endpoint,
                            "path": item["path"],
                            "role": (
                                "executable"
                                if component_id == launcher
                                else "shared_library"
                            ),
                            "sha256": item["sha256"],
                            "stat": item["stat"],
                        }
                    )
                bundles.append(
                    {
                        "bundle_id": bundle_id,
                        "endpoint": endpoint,
                        "launcher_component_id": launcher,
                        "process_role": bundle_id,
                        "required_component_ids": sorted(required_ids),
                    }
                )
                continue
            if bundle_id == "cuda_route":
                launcher = "cuda-route.runtime"
                library = "cuda_route.library"
                tokenizer = "cuda-tokenize"
                component_rows = (
                    (
                        launcher,
                        str(cuda_route_root / "operator-cuda_runtime"),
                        "cuda_runtime\n",
                        "executable",
                    ),
                    (
                        library,
                        f"{roots[bundle_id]}/{library}.so",
                        f"{library}\n",
                        "shared_library",
                    ),
                    (
                        tokenizer,
                        str(cuda_route_root / "operator-codec"),
                        "codec\n",
                        "executable",
                    ),
                )
                for component_id, path, content, role in component_rows:
                    raw = content.encode("ascii")
                    components.append(
                        {
                            "bundle_id": bundle_id,
                            "bytes": len(raw),
                            "component_id": component_id,
                            "endpoint": endpoint,
                            "path": path,
                            "role": role,
                            "sha256": sha(raw),
                            "stat": {
                                "ctime_ns": 1,
                                "device_id": 1,
                                "inode": len(component_id) + 100,
                                "mode": stat.S_IFREG
                                | (0o755 if role == "executable" else 0o644),
                                "mtime_ns": 1,
                                "size": len(raw),
                            },
                        }
                    )
                bundles.append(
                    {
                        "bundle_id": bundle_id,
                        "endpoint": endpoint,
                        "launcher_component_id": launcher,
                        "process_role": bundle_id,
                        "required_component_ids": sorted(
                            [launcher, library, tokenizer]
                        ),
                    }
                )
                continue
            if bundle_id == "op15_direct_relay":
                launcher = f"{bundle_id}.launcher"
                components.append(
                    self.component(
                        bundle_id,
                        endpoint,
                        launcher,
                        f"{roots[bundle_id]}/{launcher}",
                        "executable",
                    )
                )
                bundles.append(
                    {
                        "bundle_id": bundle_id,
                        "endpoint": endpoint,
                        "launcher_component_id": launcher,
                        "process_role": bundle_id,
                        "required_component_ids": [launcher],
                    }
                )
                continue
            launcher = (
                f"{bundle_id}.launcher"
            )
            library = f"{bundle_id}.library"
            components.extend(
                [
                    self.component(
                        bundle_id,
                        endpoint,
                        launcher,
                        f"{roots[bundle_id]}/{launcher}",
                        "executable",
                    ),
                    self.component(
                        bundle_id,
                        endpoint,
                        library,
                        f"{roots[bundle_id]}/{library}.so",
                        "shared_library",
                    ),
                ]
            )
            bundles.append(
                {
                    "bundle_id": bundle_id,
                    "endpoint": endpoint,
                    "launcher_component_id": launcher,
                    "process_role": bundle_id,
                    "required_component_ids": sorted([launcher, library]),
                }
            )
        return {
            "bundle_roots": roots,
            "bundles": bundles,
            "closure_complete": True,
            "components": components,
            "schema": "s39-cp0-r1-runtime-bundle-closure-input-v1",
        }

    def _operator(self):
        pins = {
            name: self.pin_with_stat(path)
            for name, path in self.inputs.items()
            if name != "operator_input"
        }
        for name in (
            "adb",
            "codec",
            "cuda_launcher",
            "cuda_runtime",
            "model",
            "monolithic_launcher",
            "monolithic_runtime",
            "nvidia_smi",
            "python",
            "quality_corpus",
            "ssh",
        ):
            path = (
                Path(self.runtime_inventory["bundle_roots"]["cuda_route"])
                / f"operator-{name}"
                if name in {"codec", "cuda_runtime"}
                else self.root / f"operator-{name}"
            )
            path.write_bytes(f"{name}\n".encode("ascii"))
            pins[name] = self.pin_with_stat(path)
        pins["model"] = {
            "bytes": adapter.MODEL_BYTES,
            "path": adapter.MODEL_PATH,
            "sha256": adapter.MODEL_SHA256,
            "stat": {
                "ctime_ns": 1,
                "device_id": 1,
                "inode": 1,
                "mode": stat.S_IFREG | 0o644,
                "mtime_ns": 1,
                "size": adapter.MODEL_BYTES,
            },
        }
        return {
            "controller_host": adapter.CONTROLLER_HOST,
            "directories": {
                "cuda_bundle_root": self.runtime_inventory["bundle_roots"][
                    "cuda_route"
                ],
                "joint_cwd": str(self.root),
            },
            "files": pins,
            "model_id": adapter.MODEL_ID,
            "phase": adapter.PHASE,
            "ports": {
                "adb_server": 5038,
                "cuda_monolithic": 39124,
                "cuda_route": 39125,
                "op12_stage": 39126,
                "op15_stage": 39127,
                "relay": 39128,
                "relay_tail_source": 39129,
            },
            "route_epoch": 1,
            "schema": adapter.OPERATOR_SCHEMA,
            "topology": {
                "adb_host": "127.0.0.1",
                "adb_port": 5038,
                "cuda_uuid": adapter.CUDA_UUID,
                "physical_serials": {
                    "op12": adapter.EXPECTED_PHONES["op12"]["serial"],
                    "op15": adapter.EXPECTED_PHONES["op15"]["serial"],
                },
            },
        }

    @staticmethod
    def inline(plan):
        encoded = json.dumps(
            plan,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return [
            str(adapter.USB_LAUNCHER_PATH),
            "--plan-json",
            encoded,
            "--plan-sha256",
            sha(encoded.encode("ascii")),
        ]

    def process(self, name, endpoint, bundle_id, route):
        serial = adapter.EXPECTED_PHONES[endpoint]["serial"]
        bundle = next(
            value
            for value in self.runtime_inventory["bundles"]
            if value["bundle_id"] == bundle_id
        )
        component_map = {
            value["component_id"]: value
            for value in self.runtime_inventory["components"]
        }
        components = [
            {
                key: component_map[component_id][key]
                for key in ("bytes", "component_id", "path", "sha256", "stat")
            }
            for component_id in bundle["required_component_ids"]
        ]
        launcher = component_map[bundle["launcher_component_id"]]
        adb = self.operator["files"]["adb"]
        plan = {
            "android": {
                "adb_path": adb["path"],
                "adb_port": 5038,
                "adb_selector": serial,
                "adb_sha256": adb["sha256"],
                "boot_id_source": "phase_fresh_snapshot",
                "physical_serial": serial,
                "shutdown_timeout_ms": 30000,
                "startup_timeout_ms": 120000,
            },
            "bundle_id": bundle_id,
            "components": components,
            "endpoint": endpoint,
            "launcher_component_id": bundle["launcher_component_id"],
            "mode": "android",
            "route": route,
            "schema": "s39-managed-runtime-launch-plan-v1",
        }
        return {
            "argv": self.inline(plan),
            "cwd": "/opt/s39",
            "environment": {},
            "launcher_bytes": adapter.USB_LAUNCHER_PATH.stat().st_size,
            "launcher_sha256": adapter.USB_LAUNCHER_SHA256,
            "runtime_component_ids": bundle["required_component_ids"],
            "runtime_executable_path": launcher["path"],
            "runtime_executable_sha256": launcher["sha256"],
            "shutdown_timeout_ms": 30000,
            "startup_timeout_ms": 120000,
        }

    def _phone_static(self):
        shard_path = (
            "/data/local/tmp/s39-active-warm/v1/models/"
            "qwen3-14b-q4_k_m/weights.gguf"
        )
        op12_route = {
            "devices": "GPUOpenCL",
            "driver_batch": 64,
            "driver_context": 512,
            "driver_max_prefill": 64,
            "dynamic_cut": True,
            "kind": "stagenet_worker",
            "kv_unified": True,
            "layer_end": 40,
            "layer_start": 30,
            "mode": "tailv3",
            "model_path": shard_path,
            "model_sha256": adapter.MODEL_SHA256,
            "n_gpu_layers": 999,
            "placement_cert": True,
            "port": self.operator["ports"]["op12_stage"],
            "runtime_root": self.runtime_inventory["bundle_roots"][
                "op12_stagenet"
            ],
        }
        op15_route = {
            "devices": "GPUOpenCL",
            "driver_batch": 64,
            "driver_context": 512,
            "driver_max_prefill": 64,
            "dynamic_cut": True,
            "kind": "stagenet_worker",
            "kv_unified": True,
            "layer_end": 30,
            "layer_start": 0,
            "mode": "stagenet",
            "model_path": shard_path,
            "model_sha256": adapter.MODEL_SHA256,
            "n_gpu_layers": 999,
            "placement_cert": True,
            "port": self.operator["ports"]["op15_stage"],
            "runtime_root": self.runtime_inventory["bundle_roots"][
                "op15_stagenet"
            ],
        }
        relay_route = {
            "emit_direct_frames": True,
            "head_host": adapter.UNBOUND_PHONE_NETWORK["op12"]["local_ipv4"],
            "head_port": self.operator["ports"]["op15_stage"],
            "kind": "direct_relay",
            "listen_port": self.operator["ports"]["relay"],
            "runtime_root": self.runtime_inventory["bundle_roots"][
                "op15_direct_relay"
            ],
            "tail_host": "127.0.0.1",
            "tail_port": self.operator["ports"]["op12_stage"],
            "tail_source_port": self.operator["ports"]["relay_tail_source"],
        }
        processes = {
            "op12_stagenet": self.process(
                "op12",
                "op12",
                "op12_stagenet",
                op12_route,
            ),
            "op15_direct_relay": self.process(
                "relay",
                "op15",
                "op15_direct_relay",
                relay_route,
            ),
            "op15_stagenet": self.process(
                "op15",
                "op15",
                "op15_stagenet",
                op15_route,
            ),
        }
        probes = {
            endpoint: {
                "after_argv": ["/opt/s39/probe", endpoint, "after"],
                "before_argv": ["/opt/s39/probe", endpoint, "before"],
                "cwd": "/opt/s39",
                "environment": {},
                "launcher_bytes": 1,
                "launcher_sha256": "c" * 64,
                "timeout_ms": 60000,
            }
            for endpoint in ("op12", "op15")
        }
        codec_pin = self.operator["files"]["codec"]
        codec = {
            "argv": [
                codec_pin["path"],
                "--model",
                adapter.MODEL_PATH,
                "--model-sha256",
                adapter.MODEL_SHA256,
            ],
            "cwd": "/opt/s39",
            "environment": {},
            "executable_bytes": codec_pin["bytes"],
            "executable_sha256": codec_pin["sha256"],
            "timeout_ms": 600000,
        }
        mechanism = {
            "desktop": [
                codec["argv"],
                *[["/opt/s39/desktop", str(index)] for index in range(8)],
            ],
            "op12": [
                processes["op12_stagenet"]["argv"],
                probes["op12"]["before_argv"],
                probes["op12"]["after_argv"],
            ],
            "op15": [
                processes["op15_stagenet"]["argv"],
                processes["op15_direct_relay"]["argv"],
                probes["op15"]["before_argv"],
                probes["op15"]["after_argv"],
            ],
        }
        return {
            "codec": codec,
            "expected_file_type": 15,
            "expected_max_streams": 8,
            "expected_n_batch": 64,
            "expected_n_ctx_seq": 512,
            "expected_n_embd": 5120,
            "expected_n_layer": 40,
            "expected_n_ubatch": 64,
            "mechanism_commands": mechanism,
            "phones": {
                endpoint: {
                    "device": expected["device"],
                    "executed_layers": expected["executed_layers"],
                    "expected_worker_executable_path": next(
                        component["path"]
                        for component in self.runtime_inventory["components"]
                        if component["component_id"]
                        == next(
                            bundle["launcher_component_id"]
                            for bundle in self.runtime_inventory["bundles"]
                            if bundle["bundle_id"] == f"{endpoint}_stagenet"
                        )
                    ),
                    "expected_worker_executable_sha256": next(
                        component["sha256"]
                        for component in self.runtime_inventory["components"]
                        if component["component_id"]
                        == next(
                            bundle["launcher_component_id"]
                            for bundle in self.runtime_inventory["bundles"]
                            if bundle["bundle_id"] == f"{endpoint}_stagenet"
                        )
                    ),
                    "loaded_shard_path": shard_path,
                    "loaded_shard_sha256": expected["shard_sha256"],
                    "model": expected["model"],
                    "product": expected["product"],
                    "serial": expected["serial"],
                    "stored_layers": expected["stored_layers"],
                }
                for endpoint, expected in adapter.EXPECTED_PHONES.items()
            },
            "probes": probes,
            "processes": processes,
            "relay_host": "127.0.0.1",
            "relay_port": 39128,
            "route_epoch": 1,
        }

    def _cuda_static(self, phone):
        launcher = self.operator["files"]["cuda_launcher"]
        runtime = self.operator["files"]["cuda_runtime"]
        codec_executable = self.operator["files"]["codec"]
        nvidia_executable = self.operator["files"]["nvidia_smi"]
        cuda_bundle = next(
            value
            for value in self.runtime_inventory["bundles"]
            if value["bundle_id"] == "cuda_route"
        )
        worker_argv = [
            launcher["path"],
            runtime["path"],
            "--model",
            adapter.MODEL_PATH,
            "--mode",
            "monov3",
            "--backend",
            "CUDA0",
            "--layer-start",
            "0",
            "--layer-end",
            "40",
        ]
        return {
            "codec": {
                "argv": [
                    codec_executable["path"],
                    "--model",
                    adapter.MODEL_PATH,
                    "--model-sha256",
                    adapter.MODEL_SHA256,
                ],
                "cwd": "/opt/s39",
                "environment": {},
                "executable": codec_executable,
                "timeout_ms": 600000,
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
            "io_timeout_ms": 60000,
            "mechanism_commands": phone["mechanism_commands"],
            "model_artifact": {
                **self.operator["files"]["model"],
            },
            "nvidia_smi": {
                "device_argv": [
                    nvidia_executable["path"],
                    f"--id={adapter.CUDA_UUID}",
                    "--query-gpu=name,uuid,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                "executable": nvidia_executable,
                "process_argv": [
                    nvidia_executable["path"],
                    f"--id={adapter.CUDA_UUID}",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                "timeout_ms": 60000,
            },
            "port": 39125,
            "route_epoch": 1,
            "worker": {
                "argv": worker_argv,
                "cwd": self.operator["directories"]["cuda_bundle_root"],
                "environment": {
                    "LAYERSPLIT_MEMORY_CERT": "1",
                    "LAYERSPLIT_MODEL_SHA256": adapter.MODEL_SHA256,
                    "LAYERSPLIT_PLACEMENT_CERT": "1",
                },
                "executable": launcher,
                "runtime_component_ids": cuda_bundle["required_component_ids"],
                "runtime_executable": runtime,
                "shutdown_timeout_ms": 30000,
                "startup_timeout_ms": 120000,
            },
        }

    def _runtime_static(self):
        components = [
            {key: value for key, value in component.items() if key != "stat"}
            for component in self.runtime_inventory["components"]
        ]
        captures = []
        contract = json.loads(
            self.inputs["contract"].read_text(encoding="ascii")
        )
        launchers = {
            bundle["bundle_id"]: bundle["launcher_component_id"]
            for bundle in self.runtime_inventory["bundles"]
        }
        for kind, bundle_id, endpoint in (
            ("artifact_root", "cuda_route", "cuda"),
            ("cuda_monolithic", "cuda_monolithic", "cuda"),
            ("fast_fresh_readiness", "cuda_route", "cuda"),
            ("joint_phone_cuda", "cuda_route", "cuda"),
        ):
            component_id = f"capture.{kind}"
            source = (
                contract["producer_requirements"]["source_programs"][kind]
                if kind in {"cuda_monolithic", "joint_phone_cuda"}
                else None
            )
            components.append(
                {
                    "bundle_id": bundle_id,
                    "bytes": source["bytes"] if source is not None else 1,
                    "component_id": component_id,
                    "endpoint": endpoint,
                    "path": f"{self.runtime_inventory['bundle_roots'][bundle_id]}/{component_id}",
                    "role": "executable",
                    "sha256": (
                        source["sha256"]
                        if source is not None
                        else sha(kind.encode("ascii"))
                    ),
                }
            )
            captures.append(
                {
                    "component_id": component_id,
                    "execution_mode": "SELF_CONTAINED_PHYSICAL_CAPTURE",
                    "kind": kind,
                    "nested_capture_entrypoint_component_ids": (
                        [launchers["cuda_monolithic"]]
                        if kind == "cuda_monolithic"
                        else sorted(
                            launchers[value]
                            for value in (
                                "cuda_route",
                                "op12_stagenet",
                                "op15_direct_relay",
                                "op15_stagenet",
                            )
                        )
                        if kind == "joint_phone_cuda"
                        else []
                    ),
                }
            )
        return {
            "bundle_roots": self.runtime_inventory["bundle_roots"],
            "bundles": [
                {
                    **bundle,
                    "process_role": adapter.AUTHORITY_PROCESS_ROLES[
                        bundle["bundle_id"]
                    ],
                }
                for bundle in self.runtime_inventory["bundles"]
            ],
            "capture_entrypoints": sorted(captures, key=lambda value: value["kind"]),
            "components": sorted(components, key=lambda value: value["component_id"]),
            "tokenizer_component_id": "cuda-tokenize",
        }

    def _inventory(self):
        phone = self._phone_static()
        return {
            "captured_on": {
                "controller_host": adapter.CONTROLLER_HOST,
                "cuda_uuid": adapter.CUDA_UUID,
                "phone_adb_port": 5038,
                "physical_usb_selectors": {
                    name: value["serial"]
                    for name, value in adapter.EXPECTED_PHONES.items()
                },
            },
            "inputs": {
                name: self.pin(path) for name, path in sorted(self.inputs.items())
            },
            "model_id": adapter.MODEL_ID,
            "phase": adapter.PHASE,
            "route_epoch": 1,
            "schema": adapter.INVENTORY_SCHEMA,
            "static": {
                "cuda_route": self._cuda_static(phone),
                "joint_cwd": str(self.root),
                "phone_route": phone,
                "runtime": self._runtime_static(),
            },
        }

    def rebuild(self):
        self.inventory_path.write_bytes(canonical(self.inventory))

    def build(self):
        self.rebuild()
        raw = self.inventory_path.read_bytes()
        return adapter.build_spec(self.inventory_path, sha(raw))


class AdapterTests(unittest.TestCase):
    def test_valid_inventory_builds_unbound_spec(self):
        fixture = Fixture(self)
        value = fixture.build()
        self.assertEqual(value["schema"], adapter.SPEC_SCHEMA)
        self.assertEqual(value["identity_placeholders"], adapter.UNBOUND_BOOT_IDS)
        self.assertEqual(
            value["network_placeholders"],
            adapter.UNBOUND_PHONE_NETWORK,
        )
        serialized = canonical(value)
        self.assertNotIn(b"192.0.2.12", serialized)
        self.assertIn(b"5ae7a43d", serialized)
        self.assertIn(b"3C15AU002CL00000", serialized)

    def test_build_is_byte_deterministic(self):
        fixture = Fixture(self)
        first = canonical(fixture.build())
        second = canonical(fixture.build())
        self.assertEqual(first, second)

    def test_inventory_digest_is_required(self):
        fixture = Fixture(self)
        fixture.rebuild()
        with self.assertRaisesRegex(adapter.MaterializeError, "inventory.sha256"):
            adapter.build_spec(fixture.inventory_path, "0" * 64)

    def test_noncanonical_inventory_is_rejected(self):
        fixture = Fixture(self)
        fixture.inventory_path.write_text(
            json.dumps(fixture.inventory, indent=2),
            encoding="ascii",
        )
        with self.assertRaisesRegex(adapter.MaterializeError, "E_CANONICAL"):
            adapter.build_spec(
                fixture.inventory_path,
                sha(fixture.inventory_path.read_bytes()),
            )

    def test_live_phone_ip_in_commands_is_rejected(self):
        fixture = Fixture(self)
        process = fixture.inventory["static"]["phone_route"]["processes"][
            "op15_direct_relay"
        ]
        plan = adapter._inline_plan(process["argv"], "test")
        plan["route"]["head_host"] = fixture.topology["observed"]["phones"][
            "op12"
        ]["wifi_ipv4"]
        process["argv"] = fixture.inline(plan)
        fixture.inventory["static"]["phone_route"]["mechanism_commands"]["op15"][1] = (
            process["argv"]
        )
        with self.assertRaisesRegex(adapter.MaterializeError, "E_LIVE_IDENTITY"):
            fixture.build()

    def test_wifi_adb_selector_is_rejected(self):
        fixture = Fixture(self)
        process = fixture.inventory["static"]["phone_route"]["processes"][
            "op12_stagenet"
        ]
        plan = adapter._inline_plan(process["argv"], "test")
        plan["android"]["adb_selector"] = "192.0.2.12:5555"
        process["argv"] = fixture.inline(plan)
        fixture.inventory["static"]["phone_route"]["mechanism_commands"]["op12"][0] = (
            process["argv"]
        )
        with self.assertRaisesRegex(adapter.MaterializeError, "adb_selector"):
            fixture.build()

    def test_wrong_phone_backend_is_rejected(self):
        fixture = Fixture(self)
        process = fixture.inventory["static"]["phone_route"]["processes"][
            "op15_stagenet"
        ]
        plan = adapter._inline_plan(process["argv"], "test")
        plan["route"]["devices"] = "CPU"
        process["argv"] = fixture.inline(plan)
        fixture.inventory["static"]["phone_route"]["mechanism_commands"]["op15"][0] = (
            process["argv"]
        )
        with self.assertRaisesRegex(adapter.MaterializeError, "backend"):
            fixture.build()

    def test_cut_geometry_is_rejected(self):
        fixture = Fixture(self)
        fixture.inventory["static"]["phone_route"]["phones"]["op15"][
            "executed_layers"
        ] = [0, 29]
        with self.assertRaisesRegex(adapter.MaterializeError, "executed_layers"):
            fixture.build()

    def test_shard_digest_is_rejected(self):
        fixture = Fixture(self)
        fixture.inventory["static"]["phone_route"]["phones"]["op12"][
            "loaded_shard_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(adapter.MaterializeError, "loaded_shard_sha256"):
            fixture.build()

    def test_runtime_inventory_divergence_is_rejected(self):
        fixture = Fixture(self)
        component = next(
            value
            for value in fixture.inventory["static"]["runtime"]["components"]
            if value["component_id"] == "cuda-tokenize"
        )
        component["sha256"] = "0" * 64
        with self.assertRaisesRegex(adapter.MaterializeError, "runtime.component"):
            fixture.build()

    def test_capture_closure_is_required(self):
        fixture = Fixture(self)
        fixture.inventory["static"]["runtime"]["capture_entrypoints"].pop()
        with self.assertRaisesRegex(adapter.MaterializeError, "E_CAPTURE_ENTRIES"):
            fixture.build()

    def test_operator_file_binding_is_required(self):
        fixture = Fixture(self)
        operator = copy.deepcopy(fixture.operator)
        operator["files"]["token_history"]["sha256"] = "0" * 64
        fixture.inputs["operator_input"].write_bytes(canonical(operator))
        fixture.inventory["inputs"]["operator_input"] = fixture.pin(
            fixture.inputs["operator_input"]
        )
        with self.assertRaisesRegex(adapter.MaterializeError, "operator.files"):
            fixture.build()

    def test_physical_adb_port_is_required(self):
        fixture = Fixture(self)
        fixture.inventory["captured_on"]["phone_adb_port"] = 5037
        with self.assertRaisesRegex(adapter.MaterializeError, "captured.adb_port"):
            fixture.build()

    def test_materialize_passes_exact_usb_launcher(self):
        fixture = Fixture(self)
        fixture.rebuild()
        outputs = {
            name: fixture.root / f"{name}.json"
            for name in (
                "cuda_route_launch",
                "joint_capture_plan",
                "phone_route_launch",
                "runtime_plan",
            )
        }

        spec_output = fixture.root / "spec.json"
        root = fixture.root / "root.json"
        report = fixture.root / "report.json"
        adapter.materialize(
            inventory_path=fixture.inventory_path,
            inventory_sha256=sha(fixture.inventory_path.read_bytes()),
            spec_output=spec_output,
            output_paths=outputs,
            prospective_root_output=root,
            report_output=report,
        )
        self.assertTrue(spec_output.is_file())
        self.assertTrue(all(path.is_file() for path in outputs.values()))
        self.assertEqual(
            json.loads(root.read_bytes())["status"],
            "POST_REBOOT_IDENTITY_BINDING_REQUIRED",
        )
        self.assertEqual(
            json.loads(report.read_bytes())["status"],
            "DRY_RUN_PASS_POST_REBOOT_BINDING_REQUIRED",
        )

    def test_wrong_usb_launcher_path_is_rejected(self):
        fixture = Fixture(self)
        process = fixture.inventory["static"]["phone_route"]["processes"][
            "op12_stagenet"
        ]
        process["argv"][0] = "/tmp/not-the-usb-launcher"
        with self.assertRaisesRegex(adapter.MaterializeError, "compatibility.*path"):
            adapter._require_launcher_compatible(
                fixture.build()["phone_route_static"]
            )

    def test_wrong_usb_launcher_digest_is_rejected(self):
        fixture = Fixture(self)
        process = fixture.inventory["static"]["phone_route"]["processes"][
            "op15_stagenet"
        ]
        process["launcher_sha256"] = "0" * 64
        with self.assertRaisesRegex(adapter.MaterializeError, "compatibility.*sha256"):
            adapter._require_launcher_compatible(
                fixture.build()["phone_route_static"]
            )

    def test_spec_is_accepted_by_real_frozen_originator(self):
        fixture = Fixture(self)
        spec_path = fixture.write("spec.json", fixture.build())
        outputs = {
            name: fixture.root / f"{name}.json"
            for name in (
                "cuda_route_launch",
                "joint_capture_plan",
                "phone_route_launch",
                "runtime_plan",
            )
        }
        artifacts, root, report = adapter._load_originator().build(
            spec_path=spec_path,
            output_paths=outputs,
            prospective_root_output=fixture.root / "root.json",
            report_output=fixture.root / "report.json",
        )
        self.assertEqual(set(artifacts), set(outputs))
        self.assertEqual(
            root["status"],
            "POST_REBOOT_IDENTITY_BINDING_REQUIRED",
        )
        self.assertEqual(
            report["status"],
            "DRY_RUN_PASS_POST_REBOOT_BINDING_REQUIRED",
        )

    def test_cuda_model_must_match_operator_pin(self):
        fixture = Fixture(self)
        fixture.inventory["static"]["cuda_route"]["model_artifact"]["stat"][
            "inode"
        ] += 1
        with self.assertRaisesRegex(adapter.MaterializeError, "cuda.model_artifact"):
            fixture.build()

    def test_phone_runtime_component_closure_is_required(self):
        fixture = Fixture(self)
        process = fixture.inventory["static"]["phone_route"]["processes"][
            "op12_stagenet"
        ]
        process["runtime_executable_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            adapter.MaterializeError,
            "runtime_executable_sha256",
        ):
            fixture.build()

    def test_originator_failure_removes_prospective_spec(self):
        fixture = Fixture(self)
        fixture.rebuild()
        spec_output = fixture.root / "spec.json"

        class RejectingOriginator:
            @staticmethod
            def materialize(**kwargs):
                del kwargs
                raise RuntimeError("downstream refusal")

        with mock.patch.object(
            adapter,
            "_load_originator",
            return_value=RejectingOriginator(),
        ), mock.patch.object(
            adapter,
            "_require_launcher_compatible",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "downstream refusal"):
                adapter.materialize(
                    inventory_path=fixture.inventory_path,
                    inventory_sha256=sha(fixture.inventory_path.read_bytes()),
                    spec_output=spec_output,
                    output_paths={
                        name: fixture.root / f"{name}.json"
                        for name in (
                            "cuda_route_launch",
                            "joint_capture_plan",
                            "phone_route_launch",
                            "runtime_plan",
                        )
                    },
                    prospective_root_output=fixture.root / "root.json",
                    report_output=fixture.root / "report.json",
                )
        self.assertFalse(spec_output.exists())

    def test_output_is_exclusive(self):
        fixture = Fixture(self)
        output = fixture.root / "output.json"
        adapter.write_new(output, {"value": 1})
        with self.assertRaisesRegex(adapter.MaterializeError, "E_OUTPUT_EXISTS"):
            adapter.write_new(output, {"value": 2})

    def test_cli_refuses_without_traceback(self):
        fixture = Fixture(self)
        fixture.rebuild()
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            status = adapter.main(
                [
                    "--inventory",
                    str(fixture.inventory_path),
                    "--inventory-sha256",
                    "0" * 64,
                    "--spec-output",
                    str(fixture.root / "spec.json"),
                    "--cuda-route-launch",
                    str(fixture.root / "cuda.json"),
                    "--joint-capture-plan",
                    str(fixture.root / "joint.json"),
                    "--phone-route-launch",
                    str(fixture.root / "phone.json"),
                    "--runtime-plan",
                    str(fixture.root / "runtime.json"),
                    "--prospective-root",
                    str(fixture.root / "root.json"),
                    "--dry-run-report",
                    str(fixture.root / "report.json"),
                ]
            )
        self.assertEqual(status, 2)
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
