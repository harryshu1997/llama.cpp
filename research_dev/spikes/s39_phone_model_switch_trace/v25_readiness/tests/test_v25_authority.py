#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
import stat
import sys
import unittest


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_contract_v25 as builder
import cp0_r1_evidence_v25 as authority
import v25_common as common


BOOT_CONTROLLER = "11111111-1111-1111-1111-111111111111"
BOOT_CUDA = "22222222-2222-2222-2222-222222222222"
BOOT_OP12 = "33333333-3333-3333-3333-333333333333"
BOOT_OP15 = "44444444-4444-4444-4444-444444444444"
BOOT_OP12_BEFORE = "55555555-5555-5555-5555-555555555555"
BOOT_OP15_BEFORE = "66666666-6666-6666-6666-666666666666"
OUTER_PHASE_ID = "cp0-r1-v25-a-only-test"
INNER_PHASE_ID = "cp0-r1-v24-a-only-test"


def file_stat(size: int = 10) -> dict:
    return {
        "build_id": None,
        "ctime_ns": 11,
        "device_id": 12,
        "inode": 13,
        "mode": stat.S_IFREG | 0o555,
        "mtime_ns": 14,
        "size": size,
    }


def artifact(path: str, marker: str, size: int = 10) -> dict:
    return {
        "bytes": size,
        "path": path,
        "sha256": marker * 64,
        "stat": file_stat(size),
    }


def preparation(contract: dict) -> dict:
    phones = {}
    for phone, boot in (
        ("op12", BOOT_OP12_BEFORE),
        ("op15", BOOT_OP15_BEFORE),
    ):
        serial = contract["devices"][phone]["serial"]
        phones[phone] = {
            "adb_path": "/usr/bin/adb",
            "adb_port": 5038,
            "adb_sha256": "a" * 64,
            "boot_id_before": boot,
            "disconnected_ns": 9,
            "physical_serial": serial,
            "reboot_argv": [
                "/usr/bin/adb",
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
            "host": contract["topology"]["controller_host"],
        },
        "local_python": {
            "bytes": 100,
            "path": "/usr/bin/python3.13",
            "sha256": "b" * 64,
            "stat": file_stat(100),
        },
        "phase": "A_ONLY",
        "phase_id": OUTER_PHASE_ID,
        "phones": phones,
        "schema": "s39-cp0-r1-v25-reboot-preparation-v1",
        "started_ns": 5,
    }


def discovery(contract: dict, prep: dict, prep_raw: bytes) -> dict:
    return {
        "completed_ns": 30,
        "controller": {
            "boot_id": BOOT_CONTROLLER,
            "host": contract["topology"]["controller_host"],
        },
        "cuda": {
            "boot_id": BOOT_CUDA,
            "gpu_uuid": contract["topology"]["cuda_gpu_uuid"],
            "host": contract["topology"]["cuda_host"],
            "memory_total_bytes": contract["devices"]["cuda"][
                "memory_total_bytes"
            ],
            "name": contract["devices"]["cuda"]["name"],
            "ssh_target": contract["topology"]["cuda_ssh_target"],
            "system_swap_used_bytes": 4096,
        },
        "phase": "A_ONLY",
        "phase_id": OUTER_PHASE_ID,
        "phones": {
            "op12": {
                "adb_port": 5038,
                "attested_ns": 22,
                "boot_id": BOOT_OP12,
                "device": contract["devices"]["op12"]["device"],
                "interface": "wlan0",
                "model": contract["devices"]["op12"]["model"],
                "physical_serial": contract["devices"]["op12"]["serial"],
                "product": contract["devices"]["op12"]["product"],
                "system_swap_used_bytes": 2048,
                "usb_observed_ns": 20,
                "wifi_ipv4": "192.0.2.12",
                "wifi_selector": "192.0.2.12:5555",
            },
            "op15": {
                "adb_port": 5038,
                "attested_ns": 24,
                "boot_id": BOOT_OP15,
                "device": contract["devices"]["op15"]["device"],
                "interface": "wlan1",
                "model": contract["devices"]["op15"]["model"],
                "physical_serial": contract["devices"]["op15"]["serial"],
                "product": contract["devices"]["op15"]["product"],
                "system_swap_used_bytes": 1024,
                "usb_observed_ns": 21,
                "wifi_ipv4": "192.0.2.15",
                "wifi_selector": "192.0.2.15:5555",
            },
        },
        "preparation_completed_ns": 10,
        "preparation_sha256": common.sha256_bytes(prep_raw),
        "schema": "s39-cp0-r1-v25-post-reboot-discovery-v1",
        "started_ns": 20,
    }


def managed_plan(endpoint: str = "cuda") -> dict:
    component = {
        "bytes": 10,
        "component_id": "launcher",
        "path": "/opt/launcher",
        "sha256": "a" * 64,
        "stat": file_stat(),
    }
    return {
        "_normalized": {
            "argv": ["/opt/launcher", "--serve"],
            "component_map": {"launcher": component},
            "launcher_path": "/opt/launcher",
        },
        "bundle_id": "bundle",
        "components": [component],
        "endpoint": endpoint,
    }


def process_record() -> dict:
    return {
        "boot_id": BOOT_CUDA,
        "bundle_id": "bundle",
        "endpoint": "cuda",
        "launcher_path": "/opt/launcher",
        "loaded_repo_component_ids": ["launcher"],
        "observed_ns": 101,
        "pid": 10,
        "schema": "s39-runtime-process-source-v1",
        "start_ticks": 20,
        "system_dependencies": [
            {**file_stat(), "path": "/opt/launcher"},
        ],
    }


class DiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = builder.build_contract()
        self.prep_value = preparation(self.contract)
        self.prep_raw = common.canonical_bytes(self.prep_value)
        self.prep = authority.validate_preparation(
            self.prep_value,
            self.contract,
        )
        self.value = discovery(self.contract, self.prep_value, self.prep_raw)

    def test_post_reboot_discovery_passes(self) -> None:
        result = authority.validate_discovery(
            self.value,
            self.contract,
            self.prep,
            self.prep_raw,
        )
        self.assertEqual(result["cuda"]["boot_id"], BOOT_CUDA)

    def test_unbound_preparation_root_claim_fails(self) -> None:
        value = preparation(self.contract)
        value["source_root_sha256"] = "d" * 64
        with self.assertRaises(common.EvidenceError):
            authority.validate_preparation(value, self.contract)

    def test_stale_selector_fails(self) -> None:
        self.value["phones"]["op12"]["wifi_selector"] = "192.0.2.99:5555"
        with self.assertRaises(common.EvidenceError):
            authority.validate_discovery(
                self.value, self.contract, self.prep, self.prep_raw
            )

    def test_phone_product_fails(self) -> None:
        self.value["phones"]["op15"]["product"] = "wrong"
        with self.assertRaises(common.EvidenceError):
            authority.validate_discovery(
                self.value, self.contract, self.prep, self.prep_raw
            )

    def test_gpu_name_fails(self) -> None:
        self.value["cuda"]["name"] = "different"
        with self.assertRaises(common.EvidenceError):
            authority.validate_discovery(
                self.value, self.contract, self.prep, self.prep_raw
            )

    def test_gpu_capacity_fails(self) -> None:
        self.value["cuda"]["memory_total_bytes"] -= 1
        with self.assertRaises(common.EvidenceError):
            authority.validate_discovery(
                self.value, self.contract, self.prep, self.prep_raw
            )


class AcquisitionArtifactTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, dict]:
        capture = {
            "phone_evidence": {},
            "schema": "s39-cp0-r1-v24-joint-phone-cuda-raw-v1",
        }
        capture_raw = common.canonical_bytes(capture)
        capture_path = root / "raw" / "joint-phone-cuda.json"
        capture_path.parent.mkdir()
        capture_path.write_bytes(capture_raw)
        record = {
            "bytes": len(capture_raw),
            "path": "raw/joint-phone-cuda.json",
            "role": "capture.joint_phone_cuda",
            "sha256": common.sha256_bytes(capture_raw),
        }
        acquisition = {
            "artifacts": [record],
            "phase": "A_ONLY",
            "phase_id": INNER_PHASE_ID,
            "schema": "s39-cp0-r1-a-only-acquisition-v2.4",
            "status": "RAW_CAPTURE_COMPLETE_UNEVALUATED",
        }
        acquisition_path = root / "acquisition.json"
        acquisition_path.write_bytes(common.canonical_bytes(acquisition))
        return acquisition_path, acquisition

    def test_capture_is_read_from_bound_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition_path, _ = self._fixture(root)
            value, digest = authority._read_acquisition_object_role(
                root,
                acquisition_path.name,
                "capture.joint_phone_cuda",
                INNER_PHASE_ID,
            )
            self.assertEqual(
                value["schema"],
                "s39-cp0-r1-v24-joint-phone-cuda-raw-v1",
            )
            self.assertEqual(
                digest,
                common.sha256_file(root / "raw" / "joint-phone-cuda.json"),
            )

    def test_capture_digest_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition_path, acquisition = self._fixture(root)
            acquisition["artifacts"][0]["sha256"] = "f" * 64
            acquisition_path.write_bytes(common.canonical_bytes(acquisition))
            with self.assertRaises(common.EvidenceError):
                authority._read_acquisition_object_role(
                    root,
                    acquisition_path.name,
                    "capture.joint_phone_cuda",
                    INNER_PHASE_ID,
                )

    def test_capture_role_must_exist_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition_path, acquisition = self._fixture(root)
            acquisition["artifacts"][0]["role"] = "other"
            acquisition_path.write_bytes(common.canonical_bytes(acquisition))
            with self.assertRaises(common.EvidenceError):
                authority._read_acquisition_object_role(
                    root,
                    acquisition_path.name,
                    "capture.joint_phone_cuda",
                    INNER_PHASE_ID,
            )


class V24IdentityProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = builder.build_contract()
        self.v24 = authority._load_v24_authority(self.contract)
        self.v24_contract = common.read_canonical(
            authority.S39
            / self.contract["composition"]["v24"]["contract"]["path"]
        )[0]
        mechanism = {"desktop": [], "op12": [], "op15": []}
        values = {
            "inner.cuda_route_launch": {
                "mechanism_commands": mechanism,
                "schema": "s39-cp0-r1-v24-cuda-route-launch-v1",
            },
            "inner.joint_capture_plan": {
                "mechanism_commands": mechanism,
                "schema": "s39-cp0-r1-v24-joint-capture-plan-v1",
            },
            "inner.phone_route_launch": {
                "mechanism_commands": mechanism,
                "schema": "s39-cp0-r1-v24-phone-route-launch-v1",
            },
            "inner.runtime_plan": {
                "schema": "s39-cp0-r1-runtime-bundle-plan-v2.4",
            },
            "inner.phase_lock": {
                "event_ns": 10,
                "phase_id": INNER_PHASE_ID,
                "schema": "s39-cp0-r1-phase-lock-v2.4",
            },
            "inner.preparation": {
                "schema": "s39-cp0-r1-reboot-preparation-v2.4",
            },
            "inner.prospective_root": {
                "schema": "s39-cp0-r1-v24-prospective-runtime-root-v1",
            },
        }
        raw = {
            role: common.canonical_bytes(value)
            for role, value in values.items()
        }

        def record(role: str, path: str) -> dict:
            content = raw[role]
            return {
                "bytes": len(content),
                "path": path,
                "sha256": common.sha256_bytes(content),
                "stat": file_stat(len(content)),
            }

        mechanism_sha = common.sha256_bytes(
            common.canonical_bytes(mechanism)
        )
        outputs = {
            "cuda_route_launch": record(
                "inner.cuda_route_launch",
                "/tmp/bound/cuda-route-launch.json",
            ),
            "joint_capture_plan": record(
                "inner.joint_capture_plan",
                "/tmp/bound/joint-capture-plan.json",
            ),
            "phone_route_launch": record(
                "inner.phone_route_launch",
                "/tmp/bound/phone-route-launch.json",
            ),
            "runtime_plan": record(
                "inner.runtime_plan",
                "/tmp/bound/runtime-bundle-plan.json",
            ),
        }
        receipt = {
            "completed_ns": 30,
            "desktop_identity": {
                "cuda_boot_id": BOOT_CUDA,
                "cuda_host": self.v24_contract["devices"]["cuda"]["host"],
                "cuda_ssh_target": self.v24.CUDA_SSH_TARGET,
                "cuda_uuid": self.v24_contract["devices"]["cuda"]["uuid"],
                "phone_adb_port": self.v24.PHONE_ADB_PORT,
            },
            "device_boot_ids": {
                "cuda": BOOT_CUDA,
                "op12": BOOT_OP12,
                "op15": BOOT_OP15,
            },
            "mechanism_commands_sha256": mechanism_sha,
            "outputs": outputs,
            "phase": "A_ONLY",
            "phase_id": INNER_PHASE_ID,
            "phase_lock_sha256": common.sha256_bytes(
                raw["inner.phase_lock"]
            ),
            "preparation_sha256": common.sha256_bytes(
                raw["inner.preparation"]
            ),
            "prospective_root_sha256": common.sha256_bytes(
                raw["inner.prospective_root"]
            ),
            "schema": "s39-cp0-r1-v24-identity-binding-receipt-v1",
            "started_ns": 20,
        }
        receipt_raw = common.canonical_bytes(receipt)
        bound_root = {
            "artifacts": outputs,
            "desktop_identity": receipt["desktop_identity"],
            "device_boot_ids": receipt["device_boot_ids"],
            "identity_binding_receipt": {
                "bytes": len(receipt_raw),
                "path": "/tmp/bound/identity-binding-receipt.json",
                "sha256": common.sha256_bytes(receipt_raw),
                "stat": file_stat(len(receipt_raw)),
            },
            "mechanism_commands_sha256": mechanism_sha,
            "phase": "A_ONLY",
            "phase_id": INNER_PHASE_ID,
            "phase_lock_sha256": receipt["phase_lock_sha256"],
            "preparation_sha256": receipt["preparation_sha256"],
            "prospective_root_sha256": receipt[
                "prospective_root_sha256"
            ],
            "schema": "s39-cp0-r1-v24-bound-runtime-root-v1",
        }
        bound_raw = common.canonical_bytes(bound_root)
        values.update(
            {
                "inner.bound_root": bound_root,
                "inner.identity_binding_attestation": {
                    "bound_root_sha256": common.sha256_bytes(bound_raw),
                    "identity_binding_receipt_sha256": (
                        common.sha256_bytes(receipt_raw)
                    ),
                    "schema": (
                        "s39-cp0-r1-v24-identity-binding-attestation-v1"
                    ),
                    "status": "POST_REBOOT_IDENTITY_BINDING_PASS",
                },
                "inner.identity_binding_receipt": receipt,
                "inner.identity_binding_stage_receipt": {
                    "argv": ["/usr/bin/python3", "identity_binding_v1.py"],
                    "completed_ns": 40,
                    "returncode": 0,
                    "schema": "s39-cp0-r1-v24-stage-receipt-v1",
                    "stage": "identity_binding",
                    "started_ns": 15,
                },
                "inner.orchestration_plan": {
                    "mechanism_commands_sha256": mechanism_sha,
                    "schema": (
                        "s39-cp0-r1-v24-a-only-orchestration-plan-v2"
                    ),
                },
            }
        )
        self.artifacts = {
            role: (value, common.canonical_bytes(value))
            for role, value in values.items()
        }

    def validate(self) -> dict:
        return authority._validate_v24_identity_projection(
            artifacts=self.artifacts,
            inner_lock={
                "boot_ids": {
                    "cuda": BOOT_CUDA,
                    "op12": BOOT_OP12,
                    "op15": BOOT_OP15,
                },
                "phase_id": INNER_PHASE_ID,
            },
            v24=self.v24,
            v24_contract=self.v24_contract,
        )

    def test_bound_identity_projection_passes(self) -> None:
        result = self.validate()
        self.assertEqual(result["phase_id"], INNER_PHASE_ID)

    def test_bound_launch_digest_mutation_fails(self) -> None:
        value, raw = self.artifacts["inner.cuda_route_launch"]
        del raw
        mutated = copy.deepcopy(value)
        mutated["extra"] = "changed"
        self.artifacts["inner.cuda_route_launch"] = (
            mutated,
            common.canonical_bytes(mutated),
        )
        with self.assertRaises(common.EvidenceError):
            self.validate()

    def test_attestation_root_mutation_fails(self) -> None:
        value, raw = self.artifacts["inner.identity_binding_attestation"]
        del raw
        mutated = copy.deepcopy(value)
        mutated["bound_root_sha256"] = "0" * 64
        self.artifacts["inner.identity_binding_attestation"] = (
            mutated,
            common.canonical_bytes(mutated),
        )
        with self.assertRaises(common.EvidenceError):
            self.validate()


class RuntimeIdentityTests(unittest.TestCase):
    def test_exact_process_dependencies_pass(self) -> None:
        result = authority._validate_process_record(
            process_record(),
            managed_plan(),
            BOOT_CUDA,
            100,
            "process",
        )
        self.assertEqual(result["argv"], ["/opt/launcher", "--serve"])

    def test_dependency_metadata_splice_fails(self) -> None:
        value = process_record()
        value["system_dependencies"][0]["inode"] += 1
        with self.assertRaises(common.EvidenceError):
            authority._validate_process_record(
                value,
                managed_plan(),
                BOOT_CUDA,
                100,
                "process",
            )

    def test_cuda_probe_binds_argv_and_executable(self) -> None:
        contract = builder.build_contract()
        lock = {"boot_ids": {"cuda": BOOT_CUDA}}
        process = authority._validate_process_record(
            process_record(),
            managed_plan(),
            BOOT_CUDA,
            100,
            "process",
        )
        value = {
            "boot_id": BOOT_CUDA,
            "completed_ns": 130,
            "gpu_uuid": contract["topology"]["cuda_gpu_uuid"],
            "host": contract["topology"]["cuda_host"],
            "processes": [{
                "argv": process["argv"],
                "bundle_id": process["bundle_id"],
                "executable_path": process["launcher_path"],
                "pid": process["pid"],
                "process_swap_bytes": 0,
                "start_ticks": process["start_ticks"],
            }],
            "schema": "s39-v25-remote-cuda-probe-v1",
            "ssh_target": contract["topology"]["cuda_ssh_target"],
            "started_ns": 120,
            "system_swap_used_bytes": 4096,
        }
        authority._validate_cuda_probe(
            value,
            contract,
            lock,
            110,
            [process],
            "probe",
        )
        value["processes"][0]["argv"] = ["/forged"]
        with self.assertRaises(common.EvidenceError):
            authority._validate_cuda_probe(
                value,
                contract,
                lock,
                110,
                [process],
                "probe",
            )


class PhaseBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = builder.build_contract()
        self.prep_value = preparation(self.contract)
        self.prep_raw = common.canonical_bytes(self.prep_value)
        self.prep = authority.validate_preparation(
            self.prep_value,
            self.contract,
        )
        self.discovery_raw = common.canonical_bytes(
            discovery(self.contract, self.prep_value, self.prep_raw)
        )
        self.discovery = authority.validate_discovery(
            discovery(self.contract, self.prep_value, self.prep_raw),
            self.contract,
            self.prep,
            self.prep_raw,
        )
        self.contract_raw = common.canonical_bytes(self.contract)
        self.candidate_raw = b'{"candidate":true}\n'
        self.history_raw = b'{"history":true}\n'
        self.plans = {"plan": "1" * 64}
        self.expected_plans = copy.deepcopy(self.plans)
        self.value = {
            "candidate_sha256": common.sha256_bytes(self.candidate_raw),
            "contract_sha256": common.sha256_bytes(self.contract_raw),
            "device_boot_ids": {
                "controller": BOOT_CONTROLLER,
                "cuda": BOOT_CUDA,
                "op12": BOOT_OP12,
                "op15": BOOT_OP15,
            },
            "discovery_sha256": common.sha256_bytes(self.discovery_raw),
            "event_ns": 40,
            "phase": "A_ONLY",
            "phase_id": OUTER_PHASE_ID,
            "plan_sha256s": self.plans,
            "schema": "s39-cp0-r1-v25-phase-lock-v1",
            "system_swap_baseline_bytes": {
                "cuda": 4096,
                "op12": 2048,
                "op15": 1024,
            },
            "token_history_sha256": common.sha256_bytes(self.history_raw),
            "v24_phase_id": INNER_PHASE_ID,
            "wifi_selectors": {
                "op12": "192.0.2.12:5555",
                "op15": "192.0.2.15:5555",
            },
        }

    def validate(self) -> dict:
        return authority.validate_phase_lock(
            self.value,
            self.contract,
            self.contract_raw,
            self.candidate_raw,
            self.history_raw,
            self.discovery_raw,
            self.discovery,
            self.expected_plans,
        )

    def test_nonzero_stable_swap_is_bound(self) -> None:
        self.assertEqual(self.validate()["swap"]["cuda"], 4096)

    def test_plan_digest_splice_fails(self) -> None:
        self.value["plan_sha256s"]["plan"] = "3" * 64
        with self.assertRaises(common.EvidenceError):
            self.validate()

    def test_old_phase_lock_field_fails(self) -> None:
        self.value["raw_phase_lock_artifact_sha256"] = "2" * 64
        with self.assertRaises(common.EvidenceError):
            self.validate()

    def test_raw_manifest_phase_lock_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {
                "acquisition_id": INNER_PHASE_ID,
                "phase": "A_ONLY",
                "phase_id": INNER_PHASE_ID,
                "role": "phase.lock",
            }
            row_raw = common.canonical_bytes(row)
            (root / "phase-lock.jsonl").write_bytes(row_raw)
            manifest = {
                "acquisition_started_ns": 50,
                "artifacts": [{
                    "bytes": len(row_raw),
                    "path": "phase-lock.jsonl",
                    "role": "phase.lock",
                    "sha256": common.sha256_bytes(row_raw),
                }],
                "phase": "A_ONLY",
                "phase_closed_ns": 80,
                "phase_id": INNER_PHASE_ID,
                "phase_opened_ns": 45,
            }
            manifest_raw = common.canonical_bytes(manifest)
            (root / authority.RAW_MANIFEST_NAME).write_bytes(manifest_raw)
            authority.validate_raw_run_linkage(
                raw_bundle_root=root,
                expected_manifest_sha256=common.sha256_bytes(manifest_raw),
                raw_result={
                    "phase_closed_ns": 80,
                    "phase_id": INNER_PHASE_ID,
                    "phase_opened_ns": 45,
                },
                lock={
                    "event_ns": 40,
                    "phase_id": self.value["phase_id"],
                    "v24_phase_id": INNER_PHASE_ID,
                },
                runtime={
                    "completed_ns": 90,
                    "fan_in": {
                        "remote_execution_interval": {
                            "completed_ns": 81,
                            "phase_closed_ns": 80,
                            "started_ns": 71,
                        },
                    },
                    "guard_after": {"started_ns": 82},
                    "receipts": {
                        "cuda_monolithic": {
                            "remote_execution_interval": {
                                "completed_ns": 60,
                                "started_ns": 50,
                            },
                        },
                        "joint_phone_cuda": {
                            "remote_execution_interval": {
                                "completed_ns": 70,
                                "started_ns": 61,
                            },
                        },
                    },
                    "started_ns": 41,
                },
                evidence_started_ns=30,
                evidence_completed_ns=100,
            )


def observed_artifact(value: dict) -> dict:
    return {
        **value,
        "stat": {
            "ctime_ns": 21,
            "device_id": 22,
            "inode": 23,
            "mode": stat.S_IFREG | 0o444,
            "mtime_ns": 24,
            "size": value["bytes"],
        },
    }


class RemoteHistoryAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = builder.build_contract()
        candidate_raw = common.read_regular(builder.CANDIDATE)
        history_raw = common.read_regular(builder.TOKEN_HISTORY)
        tokenizer_raw = common.read_regular(builder.TOKENIZER_PLAN)
        support = {}
        for name, record in self.contract["composition"]["history"].items():
            support[name] = {
                "bytes": record["source"]["bytes"],
                "path": record["remote_path"],
                "sha256": record["source"]["sha256"],
            }
        support["python"] = {
            "bytes": self.contract["topology"]["cuda_python_bytes"],
            "path": self.contract["topology"]["cuda_python_path"],
            "sha256": self.contract["topology"]["cuda_python_sha256"],
        }
        self.ssh = {
            "boot_id_source": "phase_fresh_snapshot",
            "identity_file_sha256": "1" * 64,
            "identity_public_key_sha256": "2" * 64,
            "known_hosts_sha256": "3" * 64,
            "remote_python_path": support["python"]["path"],
            "remote_python_sha256": support["python"]["sha256"],
            "remote_python_stat": {
                **file_stat(support["python"]["bytes"]),
            },
            "ssh_sha256": "4" * 64,
            "ssh_target": self.contract["topology"]["cuda_ssh_target"],
        }
        inputs = copy.deepcopy(self.contract["quality"]["remote_inputs"])
        command = [
            support["python"]["path"],
            "-I",
            "-c",
            authority.VALIDATOR_WRAPPER,
            str(Path(support["validator"]["path"]).parent),
            support["validator"]["path"],
            "--candidate",
            inputs["candidate"]["path"],
            "--corpus",
            inputs["corpus"]["path"],
            "--history",
            inputs["history"]["path"],
            "--tokenizer-plan",
            inputs["tokenizer_plan"]["path"],
        ]
        self.plan = {
            "command_argv": command,
            "inputs": inputs,
            "remote_cwd": self.contract["topology"]["cuda_prephase_root"],
            "schema": "s39-v25-remote-history-validation-plan-v1",
            "ssh": self.ssh,
            "support": support,
        }
        self.discovery = {
            "cuda": {
                "boot_id": BOOT_CUDA,
            },
        }
        self.receipt = {
            "boot_id": BOOT_CUDA,
            "completed_ns": 60,
            "executed_argv": command,
            "host": self.contract["topology"]["cuda_host"],
            "observed_inputs": {
                key: observed_artifact(value)
                for key, value in inputs.items()
            },
            "observed_support": {
                key: observed_artifact(value)
                for key, value in support.items()
            },
            "phase": "A_ONLY",
            "plan_sha256": common.sha256_bytes(
                common.canonical_compact(self.plan)
            ),
            "returncode": 0,
            "schema": "s39-v25-remote-history-validation-receipt-v1",
            "ssh_target": self.contract["topology"]["cuda_ssh_target"],
            "started_ns": 50,
            "stderr": "",
            "stdout": "B8_HISTORY_VALIDATE_PASS\n",
        }
        self.args = (
            candidate_raw,
            history_raw,
            tokenizer_raw,
        )

    def validate(self) -> dict:
        return authority.validate_remote_history(
            self.plan,
            self.receipt,
            self.contract,
            *self.args,
            self.discovery,
            40,
        )

    def test_remote_history_passes(self) -> None:
        self.assertEqual(self.validate()["completed_ns"], 60)

    def test_remote_input_path_substitution_fails(self) -> None:
        self.plan["inputs"]["history"]["path"] = "/tmp/history.json"
        with self.assertRaises(common.EvidenceError):
            self.validate()

    def test_remote_support_substitution_fails(self) -> None:
        self.plan["support"]["validator"]["sha256"] = "f" * 64
        with self.assertRaises(common.EvidenceError):
            self.validate()

    def test_silent_validator_fails(self) -> None:
        self.receipt["stdout"] = ""
        with self.assertRaises(common.EvidenceError):
            self.validate()

    def test_stale_boot_fails(self) -> None:
        self.receipt["boot_id"] = BOOT_OP15
        with self.assertRaises(common.EvidenceError):
            self.validate()


def phone_plan() -> dict:
    worker = artifact("/data/worker", "a")
    shard = artifact("/data/shard", "b")
    network = artifact("/data/relay", "c")
    return {
        "android": {
            "adb_path": "/usr/bin/adb",
            "adb_port": 5038,
            "adb_selector": "192.0.2.15:5555",
            "adb_sha256": "d" * 64,
        },
        "endpoint": "op15",
        "network_process": {
            "argv": ["/data/relay", "--listen", "1"],
            "artifact": network,
            "executable_path": "/data/relay",
            "role": "direct_relay",
        },
        "process": {
            "argv": ["/data/worker", "--serve"],
            "executable_path": "/data/worker",
        },
        "shard_artifact": shard,
        "telemetry": {
            "direct_peer_ipv4": "192.0.2.12",
            "direct_peer_local_port": 41000,
            "direct_peer_port": 42000,
            "interface": "wlan1",
            "local_ipv4": "192.0.2.15",
            "max_gpu_millic": 80_000,
            "min_available_bytes": 1,
        },
        "worker_artifact": worker,
    }


def phone_probe(plan: dict) -> dict:
    worker = {"pid": 11, "start_ticks": 21}
    network = {"pid": 12, "start_ticks": 22}
    return {
        "adb_path": plan["android"]["adb_path"],
        "adb_port": 5038,
        "adb_selector": plan["android"]["adb_selector"],
        "adb_sha256": plan["android"]["adb_sha256"],
        "completed_ns": 140,
        "remote": {
            "available_bytes": 1024,
            "boot_id": BOOT_OP15,
            "direct_peer": {
                "local_ipv4": "192.0.2.15",
                "local_port": 41000,
                "peer_ipv4": "192.0.2.12",
                "peer_port": 42000,
                "socket_inode": 99,
            },
            "gpu_max_millic": 70_000,
            "interface": {
                "ipv4": "192.0.2.15",
                "name": "wlan1",
                "rx_bytes": 100,
                "tx_bytes": 200,
            },
            "network_process": {
                "argv": plan["network_process"]["argv"],
                "executable_path": plan["network_process"]["executable_path"],
                "executable_sha256": plan["network_process"]["artifact"]["sha256"],
                "observed_stat": plan["network_process"]["artifact"]["stat"],
                "pid": network["pid"],
                "role": "direct_relay",
                "start_ticks": network["start_ticks"],
            },
            "physical_serial": "3C15AU002CL00000",
            "process": {
                "argv": plan["process"]["argv"],
                "executable_path": plan["process"]["executable_path"],
                "pid": worker["pid"],
                "start_ticks": worker["start_ticks"],
            },
            "process_swap_bytes": 0,
            "shard_artifact": {
                **plan["shard_artifact"],
                "observed_stat": plan["shard_artifact"]["stat"],
            },
            "system_swap_used_bytes": 0,
            "thermal_status": 0,
            "thermal_zones": [
                {"name": "gpu0", "temp_millic": 70_000},
            ],
            "worker_artifact": {
                **plan["worker_artifact"],
                "observed_stat": plan["worker_artifact"]["stat"],
            },
        },
        "schema": "s39-phone-runtime-probe-v1",
        "stage_status_source": "relay_owned_status",
        "started_ns": 130,
    }


class LegacyOuterPhoneProbeRemovalTests(unittest.TestCase):
    def test_legacy_outer_phone_probe_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_LEGACY_OUTER_PHONE_PROBE_REMOVED",
        ):
            authority._validate_probe_output(
                phone_probe(phone_plan()),
                phone_plan(),
                {},
                {},
                {},
                512,
                100,
                "probe",
            )


if __name__ == "__main__":
    unittest.main()
