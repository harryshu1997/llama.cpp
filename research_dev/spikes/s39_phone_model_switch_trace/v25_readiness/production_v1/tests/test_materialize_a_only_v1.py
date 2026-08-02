#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest


HERE = Path(__file__).resolve().parent
PRODUCTION = HERE.parent
V25 = PRODUCTION.parent
S39 = V25.parent
V24 = S39 / "v24_readiness"
V23 = S39 / "v23_readiness"
REPO = S39.parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = load("v25_materializer_test_subject", PRODUCTION / "materialize_a_only_v1.py")
common = load("v25_materializer_test_common", V25 / "v25_common.py")
previous_common = sys.modules.get("v25_common")
try:
    sys.modules["v25_common"] = common
    builder = load("v25_materializer_test_builder", V25 / "build_contract_v25.py")
    authority = load(
        "v25_materializer_test_authority",
        V25 / "cp0_r1_evidence_v25.py",
    )
finally:
    if previous_common is None:
        sys.modules.pop("v25_common", None)
    else:
        sys.modules["v25_common"] = previous_common


OUTER = "cp0-r1-v25-a-only-materializer-test"
INNER = "cp0-r1-v24-a-only-materializer-test"
BOOT_CONTROLLER = "00000000-1111-2222-3333-444444444444"
BOOT_CUDA = "10000000-1111-2222-3333-444444444444"
BOOT_OP12_BEFORE = "20000000-1111-2222-3333-444444444444"
BOOT_OP12 = "30000000-1111-2222-3333-444444444444"
BOOT_OP15_BEFORE = "40000000-1111-2222-3333-444444444444"
BOOT_OP15 = "50000000-1111-2222-3333-444444444444"


def canonical(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def fake_stat(size: int, *, executable: bool = True, inode: int = 100) -> dict:
    return {
        "build_id": None,
        "ctime_ns": 10,
        "device_id": 11,
        "inode": inode,
        "mode": stat.S_IFREG | (0o755 if executable else 0o444),
        "mtime_ns": 12,
        "size": size,
    }


def remote_artifact(
    path: str,
    raw: bytes,
    *,
    executable: bool = True,
    inode: int = 100,
) -> dict:
    return {
        "bytes": len(raw),
        "path": path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "stat": fake_stat(len(raw), executable=executable, inode=inode),
    }


class Fixture:
    def __init__(self, case: unittest.TestCase):
        temporary = tempfile.TemporaryDirectory()
        case.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.contract = builder.build_contract()
        self.contract_path = self.write("contract-v25.json", self.contract)

        python = Path(sys.executable).resolve()
        adb = Path(
            "/home/myid/zs89458/Android/Sdk/platform-tools/adb"
        ).resolve()
        ssh = Path("/usr/bin/ssh")
        ssh_keygen = Path("/usr/bin/ssh-keygen")
        identity_file = self.raw("id_ed25519", b"private-test-key\n", 0o600)
        identity_public = self.raw(
            "id_ed25519.pub",
            b"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEtest materializer\n",
            0o444,
        )
        known_hosts = self.raw(
            "known_hosts",
            b"172.20.74.85 ssh-ed25519 AAAAC3test\n",
            0o444,
        )
        self.local = {
            "adb": subject.artifact_from_path(adb, "adb"),
            "identity_file": subject.artifact_from_path(
                identity_file,
                "identity_file",
            ),
            "identity_public_key": subject.artifact_from_path(
                identity_public,
                "identity_public",
            ),
            "known_hosts": subject.artifact_from_path(known_hosts, "known_hosts"),
            "managed_launcher": subject.artifact_from_path(
                V23
                / "a_only_acquisition_driver_v1"
                / "producers_v1"
                / "managed_runtime_launcher_v1.py",
                "managed_launcher",
            ),
            "phone_guard": subject.artifact_from_path(
                V25 / "remote_phone_guard_v1.py",
                "phone_guard",
            ),
            "python": subject.artifact_from_path(python, "python"),
            "remote_cuda_capture": subject.artifact_from_path(
                V25 / "remote_cuda_capture_v1.py",
                "remote_cuda_capture",
            ),
            "remote_fan_in_contract": subject.artifact_from_path(
                V25 / "remote_fan_in_contract_v1.py",
                "remote_fan_in_contract",
            ),
            "remote_fan_in_execute": subject.artifact_from_path(
                V25 / "remote_fan_in_execute_v1.py",
                "remote_fan_in_execute",
            ),
            "remote_history": subject.artifact_from_path(
                V25 / "remote_history_validate_v1.py",
                "remote_history",
            ),
            "ssh": subject.artifact_from_path(ssh, "ssh"),
            "ssh_keygen": subject.artifact_from_path(ssh_keygen, "ssh_keygen"),
            "v25_common": subject.artifact_from_path(
                V25 / "v25_common.py",
                "v25_common",
            ),
        }
        self.preparation = self.make_preparation()
        self.preparation_path = self.write("preparation.json", self.preparation)
        preparation_raw = self.preparation_path.read_bytes()
        self.discovery = self.make_discovery(preparation_raw)
        self.discovery_path = self.write("discovery.json", self.discovery)
        self.v24_inputs = self.make_v24_inputs()
        self.pre_inputs = self.make_pre_inputs()
        self.inventory = self.make_inventory()
        self.remote = self.make_remote_artifacts()
        self.identity = self.make_identity()
        self.identity_path = self.write("identity.json", self.identity)
        self.inventory_path = self.write("inventory.json", self.inventory)

    def raw(self, name: str, raw: bytes, mode: int = 0o644) -> Path:
        path = self.root / name
        path.write_bytes(raw)
        path.chmod(mode)
        return path

    def write(self, name: str, value) -> Path:
        return self.raw(name, canonical(value))

    def pin(self, path: Path) -> dict:
        return subject.artifact_from_path(path, str(path))

    def make_preparation(self) -> dict:
        phones = {}
        for phone, before in (
            ("op12", BOOT_OP12_BEFORE),
            ("op15", BOOT_OP15_BEFORE),
        ):
            serial = self.contract["devices"][phone]["serial"]
            adb = self.local["adb"]
            phones[phone] = {
                "adb_path": adb["path"],
                "adb_port": 5038,
                "adb_sha256": adb["sha256"],
                "boot_id_before": before,
                "disconnected_ns": 9,
                "physical_serial": serial,
                "reboot_argv": [
                    adb["path"],
                    "-P",
                    "5038",
                    "-s",
                    serial,
                    "reboot",
                ],
                "reboot_returncode": 0,
                "requested_ns": 8,
            }
        return {
            "completed_ns": 10,
            "controller": {
                "boot_id": BOOT_CONTROLLER,
                "host": self.contract["topology"]["controller_host"],
            },
            "local_python": self.local["python"],
            "phase": "A_ONLY",
            "phase_id": OUTER,
            "phones": phones,
            "schema": "s39-cp0-r1-v25-reboot-preparation-v1",
            "started_ns": 5,
        }

    def make_discovery(self, preparation_raw: bytes) -> dict:
        return {
            "completed_ns": 30,
            "controller": {
                "boot_id": BOOT_CONTROLLER,
                "host": self.contract["topology"]["controller_host"],
            },
            "cuda": {
                "boot_id": BOOT_CUDA,
                "gpu_uuid": self.contract["topology"]["cuda_gpu_uuid"],
                "host": self.contract["topology"]["cuda_host"],
                "memory_total_bytes": self.contract["devices"]["cuda"][
                    "memory_total_bytes"
                ],
                "name": self.contract["devices"]["cuda"]["name"],
                "ssh_target": self.contract["topology"]["cuda_ssh_target"],
                "system_swap_used_bytes": 4096,
            },
            "phase": "A_ONLY",
            "phase_id": OUTER,
            "phones": {
                "op12": self.phone_discovery("op12", BOOT_OP12, "192.0.2.12", 22),
                "op15": self.phone_discovery("op15", BOOT_OP15, "192.0.2.15", 24),
            },
            "preparation_completed_ns": 10,
            "preparation_sha256": hashlib.sha256(preparation_raw).hexdigest(),
            "schema": "s39-cp0-r1-v25-post-reboot-discovery-v1",
            "started_ns": 20,
        }

    def phone_discovery(
        self,
        phone: str,
        boot_id: str,
        ipv4: str,
        attested: int,
    ) -> dict:
        expected = self.contract["devices"][phone]
        return {
            "adb_port": 5038,
            "attested_ns": attested,
            "boot_id": boot_id,
            "device": expected["device"],
            "interface": "wlan0",
            "model": expected["model"],
            "physical_serial": expected["serial"],
            "product": expected["product"],
            "system_swap_used_bytes": 1024,
            "usb_observed_ns": 20,
            "wifi_ipv4": ipv4,
            "wifi_selector": f"{ipv4}:5555",
        }

    def make_v24_inputs(self) -> dict:
        actual = {
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
            "contract_v25": self.contract_path,
        }
        result = {name: self.pin(path) for name, path in actual.items()}
        mono = json.loads(actual["cuda_monolithic_launch"].read_text(encoding="ascii"))
        model_sha = mono["model_sha256"]
        mechanism = {"desktop": [], "op12": [], "op15": []}
        minimal = {
            "artifact_root": {
                "schema": "s39-cp0-r1-artifact-root-v2.4",
            },
            "bound_root": {
                "schema": "s39-cp0-r1-v24-bound-runtime-root-v1",
            },
            "cuda_route_launch": {
                "schema": "s39-cp0-r1-v24-cuda-route-launch-v1",
            },
            "fresh_readiness": {
                "schema": "s39-cp0-r1-fast-fresh-readiness-v2.4",
            },
            "identity_binding_attestation": {
                "bound_root_sha256": "a" * 64,
                "identity_binding_receipt_sha256": "b" * 64,
                "schema": "s39-cp0-r1-v24-identity-binding-attestation-v1",
                "status": "POST_REBOOT_IDENTITY_BINDING_PASS",
            },
            "identity_binding_receipt": {
                "schema": "s39-cp0-r1-v24-identity-binding-receipt-v1",
            },
            "identity_binding_stage_receipt": {
                "schema": "s39-cp0-r1-v24-stage-receipt-v1",
            },
            "joint_capture_plan": {
                "schema": "s39-cp0-r1-v24-joint-capture-plan-v1",
            },
            "orchestration_plan": {
                "schema": "s39-cp0-r1-v24-a-only-orchestration-plan-v2",
            },
            "phase_lock": {
                "schema": "s39-cp0-r1-phase-lock-v2.4",
            },
            "phone_route_launch": {
                "mechanism_commands": mechanism,
                "schema": "s39-cp0-r1-v24-phone-route-launch-v1",
            },
            "preparation": {
                "schema": "s39-cp0-r1-reboot-preparation-v2.4",
            },
            "prospective_root": {
                "schema": "s39-cp0-r1-v24-prospective-runtime-root-v1",
            },
            "runtime_plan": {
                "schema": "s39-cp0-r1-runtime-bundle-plan-v2.4",
            },
        }
        del model_sha
        for name, value in minimal.items():
            result[name] = self.pin(self.write(f"v24-{name}.json", value))
        return result

    def make_pre_inputs(self) -> dict:
        root = self.root / "pre" / "raw"
        root.mkdir(parents=True)
        result = {}
        for name, filename in subject.PRE_INPUT_NAMES.items():
            result[name] = self.pin(
                self.raw(f"pre/raw/{filename}", f"{name}\n".encode("ascii"))
            )
        return result

    def make_inventory(self) -> dict:
        return {
            "acquisition_started_ns": 31,
            "desktop_forbidden_listen_ports": [39312, 39315],
            "desktop_forbidden_processes": [
                {"executable_path": "/opt/s39/op12-worker", "sha256": "a" * 64},
                {"executable_path": "/opt/s39/op15-worker", "sha256": "b" * 64},
            ],
            "local_forward_ports": {
                "cuda_monolithic": {"local": 49101, "remote": 49201},
                "joint_phone_cuda": {"local": 49102, "remote": 49202},
                "remote_fan_in": {"local": 49103, "remote": 49203},
            },
            "output_paths": {
                "cuda_monolithic": "/home/zhihao/s39-v25/run/cuda.json",
                "fan_in_acquisition": "/home/zhihao/s39-v25/run/acquisition.json",
                "fan_in_bundle_root": "/home/zhihao/s39-v25/run/bundle",
                "fan_in_runtime": "/home/zhihao/s39-v25/run/runtime.json",
                "joint_phone_cuda": "/home/zhihao/s39-v25/run/joint.json",
                "remote_root": "/home/zhihao/llama.cpp-s40",
            },
            "phase": "A_ONLY",
            "phase_id": OUTER,
            "phone_forbidden_listen_ports": {
                "op12": [39312],
                "op15": [39315],
            },
            "phone_forbidden_processes": {
                "op12": [
                    {
                        "executable_path": "/data/local/tmp/op12-worker",
                        "sha256": "c" * 64,
                    }
                ],
                "op15": [
                    {
                        "executable_path": "/data/local/tmp/op15-worker",
                        "sha256": "d" * 64,
                    }
                ],
            },
            "pre_inputs": self.pre_inputs,
            "schema": subject.INVENTORY_SCHEMA,
            "ssh": {
                "connect_timeout_s": 10,
                "identity_public_key_fingerprint": "SHA256:test",
                "shutdown_timeout_ms": 30000,
                "startup_timeout_ms": 300000,
            },
            "v24_inputs": self.v24_inputs,
            "v24_phase_id": INNER,
        }

    def make_remote_artifacts(self) -> dict:
        def source(section: str, name: str, inode: int) -> dict:
            pin = self.contract["composition"][section][name]
            raw = (S39 / pin["path"]).read_bytes()
            return remote_artifact(
                str(
                    Path(
                        "/home/zhihao/llama.cpp-s40/research_dev/spikes/"
                        "s39_phone_model_switch_trace"
                    )
                    / pin["path"]
                ),
                raw,
                executable=False,
                inode=inode,
            )

        remote = {
            "adb": remote_artifact("/usr/bin/adb", b"remote-adb\n", inode=201),
            "cuda_monolithic_producer": source(
                "v24",
                "cuda_monolithic_producer",
                202,
            ),
            "fan_in_authority": source(
                "v24",
                "authority",
                203,
            ),
            "fan_in_producer": source(
                "v24",
                "remote_fan_in",
                204,
            ),
            "joint_phone_cuda_producer": source(
                "v24",
                "joint_phone_cuda_producer",
                205,
            ),
            "nvidia_smi": remote_artifact(
                "/usr/bin/nvidia-smi",
                b"nvidia-smi\n",
                inode=206,
            ),
            "phone_guard": source(
                "v25",
                "remote_phone_guard",
                207,
            ),
            "production_common": source(
                "v24",
                "production_common",
                208,
            ),
            "python": {
                "bytes": self.contract["topology"]["cuda_python_bytes"],
                "path": self.contract["topology"]["cuda_python_path"],
                "sha256": self.contract["topology"]["cuda_python_sha256"],
                "stat": fake_stat(
                    self.contract["topology"]["cuda_python_bytes"],
                    inode=209,
                ),
            },
            "v24_common": source(
                "v24",
                "common",
                210,
            ),
            "v24_contract_builder": source(
                "v24",
                "contract_builder",
                211,
            ),
        }
        policy = self.policy_value(remote)
        policy_raw = canonical(policy)
        remote["phone_guard_policy"] = remote_artifact(
            "/home/zhihao/s39-v25/remote-phone-policy.json",
            policy_raw,
            executable=False,
            inode=212,
        )
        return remote

    def policy_value(self, remote: dict) -> dict:
        return {
            "adb_server_port": 5038,
            "adb_server_process": {
                "argv": [
                    "adb",
                    "-L",
                    "tcp:5038",
                    "fork-server",
                    "server",
                    "--reply-fd",
                    "4",
                ],
                "boot_id": BOOT_CUDA,
                "executable_path": remote["adb"]["path"],
                "listen_host": "127.0.0.1",
                "listen_port": 5038,
                "pid": 88,
                "start_ticks": 99,
            },
            "desktop_forbidden_listen_ports": self.inventory[
                "desktop_forbidden_listen_ports"
            ],
            "desktop_forbidden_processes": self.inventory[
                "desktop_forbidden_processes"
            ],
            "forbid_adb_forward_for_selectors": True,
            "gpu_uuid": subject.GPU_UUID,
            "inner_phase_id": INNER,
            "outer_phase_id": OUTER,
            "phase": "A_ONLY",
            "phones": {
                phone: {
                    "boot_id": self.discovery["phones"][phone]["boot_id"],
                    "forbidden_listen_ports": self.inventory[
                        "phone_forbidden_listen_ports"
                    ][phone],
                    "forbidden_processes": self.inventory[
                        "phone_forbidden_processes"
                    ][phone],
                    "interface": self.discovery["phones"][phone]["interface"],
                    "physical_serial": self.discovery["phones"][phone][
                        "physical_serial"
                    ],
                    "wifi_ipv4": self.discovery["phones"][phone]["wifi_ipv4"],
                    "wifi_selector": self.discovery["phones"][phone][
                        "wifi_selector"
                    ],
                }
                for phone in ("op12", "op15")
            },
            "remote_artifacts": {
                "adb": remote["adb"],
                "helper": remote["phone_guard"],
                "python": remote["python"],
            },
            "rtx_boot_id": BOOT_CUDA,
            "schema": "s39-v25-remote-phone-guard-policy-v1",
        }

    def make_identity(self) -> dict:
        remote_inputs = {}
        for index, role in enumerate(sorted(subject.FAN_IN_INPUT_ROLES), 300):
            if role.startswith("pre."):
                local = self.pre_inputs[role.removeprefix("pre.")]
                path = (
                    Path("/home/zhihao/s39-v25-a-only/pre/raw")
                    / subject.PRE_INPUT_NAMES[role.removeprefix("pre.")]
                )
            else:
                local = self.v24_inputs[role]
                path = Path("/home/zhihao/s39-v25-a-only/input") / (
                    role + ".json"
                )
            remote_inputs[role] = {
                "bytes": local["bytes"],
                "path": str(path),
                "sha256": local["sha256"],
                "stat": fake_stat(
                    local["bytes"],
                    executable=False,
                    inode=index,
                ),
            }
        return {
            "adb_server_process": self.policy_value(self.remote)[
                "adb_server_process"
            ],
            "completed_ns": 40,
            "controller_boot_id": BOOT_CONTROLLER,
            "discovery_sha256": hashlib.sha256(
                self.discovery_path.read_bytes()
            ).hexdigest(),
            "gpu_uuid": subject.GPU_UUID,
            "local_artifacts": self.local,
            "outer_phase_id": OUTER,
            "phase": "A_ONLY",
            "remote_artifacts": self.remote,
            "remote_inputs": remote_inputs,
            "rtx_boot_id": BOOT_CUDA,
            "schema": subject.IDENTITY_SCHEMA,
            "started_ns": 31,
            "v24_phase_id": INNER,
        }


class MaterializerTests(unittest.TestCase):
    def authority_fixture(self, fixture, output, root):
        artifacts = {}
        for row in root["artifacts"]:
            raw = (output / row["path"]).read_bytes()
            artifacts[row["role"]] = (
                json.loads(raw.decode("ascii")),
                raw,
            )
        for role, path in (
            ("phase.fresh_identity", fixture.identity_path),
            ("phase.inventory", fixture.inventory_path),
        ):
            raw = path.read_bytes()
            artifacts[role] = (json.loads(raw.decode("ascii")), raw)
        root_raw = (output / "MATERIALIZATION.json").read_bytes()
        artifacts["phase.materialization"] = (root, root_raw)
        discovery_raw = fixture.discovery_path.read_bytes()
        return artifacts, discovery_raw

    def test_phase_suffix_is_exact(self):
        self.assertEqual(subject.phase_suffix(OUTER, INNER), "materializer-test")
        with self.assertRaisesRegex(subject.MaterializeError, "E_PHASE_LINK"):
            subject.phase_suffix(OUTER, INNER + "-other")

    def test_identity_requires_fresh_discovery_binding(self):
        fixture = Fixture(self)
        value = copy.deepcopy(fixture.identity)
        value["discovery_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            subject.MaterializeError,
            "identity.discovery",
        ):
            subject.validate_identity(
                value,
                canonical(value),
                fixture.discovery,
                fixture.discovery_path.read_bytes(),
                fixture.contract,
            )

    def test_identity_rejects_client_launch_argv(self):
        fixture = Fixture(self)
        value = copy.deepcopy(fixture.identity)
        value["adb_server_process"]["argv"] = [
            value["remote_artifacts"]["adb"]["path"],
            "-P",
            "5038",
            "nodaemon",
            "server",
        ]
        with self.assertRaisesRegex(subject.MaterializeError, "E_ADB_SERVER_ARGV"):
            subject.validate_identity(
                value,
                canonical(value),
                fixture.discovery,
                fixture.discovery_path.read_bytes(),
                fixture.contract,
            )

    def test_identity_rejects_remote_source_drift(self):
        fixture = Fixture(self)
        value = copy.deepcopy(fixture.identity)
        value["remote_artifacts"]["cuda_monolithic_producer"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            subject.MaterializeError,
            "identity.remote.cuda_monolithic_producer.content",
        ):
            subject.validate_identity(
                value,
                canonical(value),
                fixture.discovery,
                fixture.discovery_path.read_bytes(),
                fixture.contract,
            )

    def test_identity_python_must_match_reboot_preparation(self):
        fixture = Fixture(self)
        alternate = fixture.root / "python-alternate"
        alternate.write_bytes(b"#!/bin/sh\nexit 0\n")
        alternate.chmod(0o755)
        value = copy.deepcopy(fixture.identity)
        value["local_artifacts"]["python"] = fixture.pin(alternate)
        identity_path = fixture.write("alternate-python-identity.json", value)
        with self.assertRaisesRegex(
            subject.MaterializeError,
            "identity.local.python.preparation",
        ):
            subject.materialize(
                inventory_path=fixture.inventory_path,
                inventory_sha256=hashlib.sha256(
                    fixture.inventory_path.read_bytes()
                ).hexdigest(),
                preparation_path=fixture.preparation_path,
                discovery_path=fixture.discovery_path,
                identity_path=identity_path,
                output_root=fixture.root / "alternate-python-output",
            )

    def test_inventory_rejects_forward_port_alias(self):
        fixture = Fixture(self)
        value = copy.deepcopy(fixture.inventory)
        value["local_forward_ports"]["joint_phone_cuda"]["local"] = 49101
        with self.assertRaisesRegex(subject.MaterializeError, "E_FORWARD_PORTS"):
            subject.validate_inventory(value, fixture.contract)

    def test_inventory_rejects_output_alias(self):
        fixture = Fixture(self)
        value = copy.deepcopy(fixture.inventory)
        value["output_paths"]["fan_in_runtime"] = value["output_paths"][
            "fan_in_acquisition"
        ]
        with self.assertRaisesRegex(subject.MaterializeError, "E_OUTPUT_PATH_REUSE"):
            subject.validate_inventory(value, fixture.contract)

    def test_controller_input_stat_is_load_bearing(self):
        fixture = Fixture(self)
        value = copy.deepcopy(fixture.inventory)
        value["v24_inputs"]["orchestration_plan"]["stat"]["mtime_ns"] += 1
        inventory_path = fixture.write("bad-input-stat-inventory.json", value)
        with self.assertRaisesRegex(
            subject.MaterializeError,
            "v24.orchestration_plan.identity",
        ):
            subject.materialize(
                inventory_path=inventory_path,
                inventory_sha256=hashlib.sha256(
                    inventory_path.read_bytes()
                ).hexdigest(),
                preparation_path=fixture.preparation_path,
                discovery_path=fixture.discovery_path,
                identity_path=fixture.identity_path,
                output_root=fixture.root / "bad-input-stat-output",
            )

    def test_pre_input_bytes_are_reopened(self):
        fixture = Fixture(self)
        Path(fixture.pre_inputs["route_lock"]["path"]).write_bytes(b"changed\n")
        with self.assertRaisesRegex(subject.MaterializeError, "pre.route_lock"):
            subject.materialize(
                inventory_path=fixture.inventory_path,
                inventory_sha256=hashlib.sha256(
                    fixture.inventory_path.read_bytes()
                ).hexdigest(),
                preparation_path=fixture.preparation_path,
                discovery_path=fixture.discovery_path,
                identity_path=fixture.identity_path,
                output_root=fixture.root / "bad-pre-output",
            )

    def test_attestation_cannot_replace_stage_receipt(self):
        fixture = Fixture(self)
        value = copy.deepcopy(fixture.inventory)
        attestation = json.loads(
            Path(
                value["v24_inputs"]["identity_binding_attestation"]["path"]
            ).read_text(encoding="ascii")
        )
        substituted = fixture.write("substituted-stage-receipt.json", attestation)
        value["v24_inputs"]["identity_binding_stage_receipt"] = fixture.pin(
            substituted
        )
        inventory_path = fixture.write("bad-stage-receipt-inventory.json", value)
        with self.assertRaisesRegex(
            subject.MaterializeError,
            "v24.identity_binding_stage_receipt.schema",
        ):
            subject.materialize(
                inventory_path=inventory_path,
                inventory_sha256=hashlib.sha256(
                    inventory_path.read_bytes()
                ).hexdigest(),
                preparation_path=fixture.preparation_path,
                discovery_path=fixture.discovery_path,
                identity_path=fixture.identity_path,
                output_root=fixture.root / "bad-stage-receipt-output",
            )

    def test_remote_input_must_match_controller_source(self):
        fixture = Fixture(self)
        value = copy.deepcopy(fixture.identity)
        value["remote_inputs"]["orchestration_plan"]["sha256"] = "0" * 64
        identity_path = fixture.write("bad-remote-input.json", value)
        with self.assertRaisesRegex(
            subject.MaterializeError,
            "identity.remote_inputs.orchestration_plan.content",
        ):
            subject.materialize(
                inventory_path=fixture.inventory_path,
                inventory_sha256=hashlib.sha256(
                    fixture.inventory_path.read_bytes()
                ).hexdigest(),
                preparation_path=fixture.preparation_path,
                discovery_path=fixture.discovery_path,
                identity_path=identity_path,
                output_root=fixture.root / "bad-remote-input-output",
            )

    def test_remote_policy_content_is_bound(self):
        fixture = Fixture(self)
        value = copy.deepcopy(fixture.identity)
        value["remote_artifacts"]["phone_guard_policy"]["sha256"] = "0" * 64
        path = fixture.write("bad-identity.json", value)
        output = fixture.root / "bad-output"
        with self.assertRaisesRegex(
            subject.MaterializeError,
            "remote_policy.content",
        ):
            subject.materialize(
                inventory_path=fixture.inventory_path,
                inventory_sha256=hashlib.sha256(
                    fixture.inventory_path.read_bytes()
                ).hexdigest(),
                preparation_path=fixture.preparation_path,
                discovery_path=fixture.discovery_path,
                identity_path=path,
                output_root=output,
            )
        self.assertFalse(output.exists())

    def test_materialization_is_complete_and_binds_inner_inputs(self):
        fixture = Fixture(self)
        output = fixture.root / "output"
        root = subject.materialize(
            inventory_path=fixture.inventory_path,
            inventory_sha256=hashlib.sha256(
                fixture.inventory_path.read_bytes()
            ).hexdigest(),
            preparation_path=fixture.preparation_path,
            discovery_path=fixture.discovery_path,
            identity_path=fixture.identity_path,
            output_root=output,
        )
        self.assertEqual(root["fan_in_materialized"], True)
        self.assertEqual(
            root["status"],
            "A_ONLY_PLANS_MATERIALIZED_NO_HARDWARE_RUN",
        )
        self.assertLess(root["started_ns"], root["completed_ns"])
        self.assertLessEqual(
            fixture.identity["completed_ns"],
            root["started_ns"],
        )
        self.assertIn("remote_fan_in", root["stages"])
        roles = {row["role"] for row in root["artifacts"]}
        self.assertTrue({
            "inner.artifact_root",
            "inner.bound_root",
            "inner.cuda_route_launch",
            "inner.fresh_readiness",
            "inner.identity_binding_attestation",
            "inner.identity_binding_receipt",
            "inner.identity_binding_stage_receipt",
            "inner.joint_capture_plan",
            "inner.orchestration_plan",
            "inner.phase_lock",
            "inner.phone_route_launch",
            "inner.preparation",
            "inner.prospective_root",
            "inner.runtime_plan",
            "phase.discovery",
            "phase.preparation",
            "plan.managed.remote_fan_in",
            "plan.remote_fan_in",
        }.issubset(roles))
        self.assertTrue((output / "MATERIALIZATION.json").is_file())
        command_plan_sha256 = fixture.v24_inputs["orchestration_plan"]["sha256"]
        mono = json.loads(
            (output / "wrapper-cuda_monolithic.json").read_text(encoding="ascii")
        )
        joint = json.loads(
            (output / "wrapper-joint_phone_cuda.json").read_text(encoding="ascii")
        )
        self.assertEqual(
            mono["producer_argv"][mono["producer_argv"].index("--plan") + 1],
            command_plan_sha256,
        )
        self.assertEqual(
            joint["producer_argv"][
                joint["producer_argv"].index("--command-plan-sha256") + 1
            ],
            command_plan_sha256,
        )
        self.assertEqual(
            root["stages"]["remote_history"]["argv"][1],
            fixture.local["remote_history"]["path"],
        )
        self.assertEqual(
            root["stages"]["remote_fan_in"]["argv"][1],
            fixture.local["remote_fan_in_execute"]["path"],
        )
        for role in ("cuda_monolithic", "joint_phone_cuda"):
            argv = root["stages"][role]["argv"]
            self.assertEqual(
                argv[argv.index("--remote-boot-id") + 1],
                fixture.identity["rtx_boot_id"],
            )
            self.assertEqual(
                argv[argv.index("--output") + 1],
                str(output / f"{role.replace('_', '-')}.json"),
            )
            self.assertEqual(
                argv[argv.index("--receipt") + 1],
                root["stages"][role]["expected_output"],
            )
        fan = json.loads(
            (output / "remote-fan-in-plan.json").read_text(encoding="ascii")
        )
        self.assertEqual(fan["local_common"], fixture.local["v25_common"])
        self.assertEqual(
            fan["input_artifacts"]["orchestration_plan"],
            fixture.identity["remote_inputs"]["orchestration_plan"],
        )
        self.assertNotIn(
            fan["input_artifacts"]["orchestration_plan"]["path"],
            fan["producer_argv"],
        )
        managed_fan = json.loads(
            (output / "managed-remote_fan_in.json").read_text(encoding="ascii")
        )
        component_paths = {row["path"] for row in managed_fan["components"]}
        self.assertIn(
            fan["input_artifacts"]["orchestration_plan"]["path"],
            component_paths,
        )
        self.assertNotIn(fan["local_common"]["path"], component_paths)

    def test_existing_output_fails_closed(self):
        fixture = Fixture(self)
        output = fixture.root / "existing"
        output.mkdir()
        with self.assertRaisesRegex(subject.MaterializeError, "E_OUTPUT_EXISTS"):
            subject.materialize(
                inventory_path=fixture.inventory_path,
                inventory_sha256=hashlib.sha256(
                    fixture.inventory_path.read_bytes()
                ).hexdigest(),
                preparation_path=fixture.preparation_path,
                discovery_path=fixture.discovery_path,
                identity_path=fixture.identity_path,
                output_root=output,
            )

    def test_authority_revalidates_materialization(self):
        fixture = Fixture(self)
        output = fixture.root / "authority-pass"
        root = subject.materialize(
            inventory_path=fixture.inventory_path,
            inventory_sha256=hashlib.sha256(
                fixture.inventory_path.read_bytes()
            ).hexdigest(),
            preparation_path=fixture.preparation_path,
            discovery_path=fixture.discovery_path,
            identity_path=fixture.identity_path,
            output_root=output,
        )
        artifacts, discovery_raw = self.authority_fixture(
            fixture,
            output,
            root,
        )
        derived = authority.validate_materialized_prelock(
            artifacts=artifacts,
            contract=fixture.contract,
            discovery_raw=discovery_raw,
            discovery_value=fixture.discovery,
        )
        self.assertEqual(derived["inventory"]["phase_id"], OUTER)

    def test_authority_rejects_materialized_artifact_mutation(self):
        fixture = Fixture(self)
        output = fixture.root / "authority-mutation"
        root = subject.materialize(
            inventory_path=fixture.inventory_path,
            inventory_sha256=hashlib.sha256(
                fixture.inventory_path.read_bytes()
            ).hexdigest(),
            preparation_path=fixture.preparation_path,
            discovery_path=fixture.discovery_path,
            identity_path=fixture.identity_path,
            output_root=output,
        )
        artifacts, discovery_raw = self.authority_fixture(
            fixture,
            output,
            root,
        )
        root["artifacts"][0]["sha256"] = "0" * 64
        with self.assertRaises(common.EvidenceError):
            authority.validate_materialized_prelock(
                artifacts=artifacts,
                contract=fixture.contract,
                discovery_raw=discovery_raw,
                discovery_value=fixture.discovery,
            )

    def test_authority_rejects_materialized_stage_argv_mutation(self):
        fixture = Fixture(self)
        output = fixture.root / "authority-argv-mutation"
        root = subject.materialize(
            inventory_path=fixture.inventory_path,
            inventory_sha256=hashlib.sha256(
                fixture.inventory_path.read_bytes()
            ).hexdigest(),
            preparation_path=fixture.preparation_path,
            discovery_path=fixture.discovery_path,
            identity_path=fixture.identity_path,
            output_root=output,
        )
        artifacts, discovery_raw = self.authority_fixture(
            fixture,
            output,
            root,
        )
        root["stages"]["cuda_monolithic"]["argv"][-1] = (
            "RUN-S39-V25-REMOTE-CUDA-ALTERED"
        )
        with self.assertRaises(common.EvidenceError):
            authority.validate_materialized_prelock(
                artifacts=artifacts,
                contract=fixture.contract,
                discovery_raw=discovery_raw,
                discovery_value=fixture.discovery,
            )


if __name__ == "__main__":
    unittest.main()
