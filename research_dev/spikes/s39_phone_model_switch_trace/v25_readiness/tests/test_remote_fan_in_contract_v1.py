#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest


HERE = Path(__file__).resolve().parents[1]


def load_source(name: str, path: Path, injected: dict[str, object] | None = None):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    previous = {key: sys.modules.get(key) for key in (injected or {})}
    try:
        if injected:
            sys.modules.update(injected)
        raw = path.read_bytes()
        exec(compile(raw, str(path), "exec"), module.__dict__)
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


common = load_source("v25_common", HERE / "v25_common.py")
fan_in = load_source(
    "remote_fan_in_contract_v1",
    HERE / "remote_fan_in_contract_v1.py",
    {"v25_common": common},
)

OUTER = "cp0-r1-v25-a-only-test"
INNER = "cp0-r1-v24-a-only-test"
CONTROLLER_BOOT = "11111111-2222-3333-4444-555555555555"
RTX_BOOT = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
GPU = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"


def stat_row(size: int, inode: int = 3, executable: bool = True) -> dict:
    return {
        "build_id": None,
        "ctime_ns": 1,
        "device_id": 2,
        "inode": inode,
        "mode": stat.S_IFREG | (0o555 if executable else 0o444),
        "mtime_ns": 4,
        "size": size,
    }


def artifact(
    path: str,
    marker: str,
    size: int = 10,
    inode: int = 3,
) -> dict:
    return {
        "bytes": size,
        "path": path,
        "sha256": marker * 64,
        "stat": stat_row(size, inode),
    }


def wrapper_plan() -> dict:
    inputs = {
        role: artifact(f"/remote/input/{index}", str(index % 10), inode=100 + index)
        for index, role in enumerate(sorted(fan_in.INPUT_ROLES), 1)
    }
    for role, name in fan_in.PRE_INPUT_NAMES.items():
        inputs[role]["path"] = f"/remote/pre/raw/{name}"
    sources = {
        role: artifact(
            f"/remote/source/{index}-{role}.py",
            "abcdef"[index],
            inode=200 + index,
        )
        for index, role in enumerate(sorted(fan_in.SOURCE_ROLES))
    }
    value = {
        "acquisition_started_ns": 100,
        "capture_input_paths": {
            "cuda_monolithic": "/remote/capture/cuda-monolithic.json",
            "joint_phone_cuda": "/remote/capture/joint-phone-cuda.json",
        },
        "contract_validator": artifact("/repo/fan-contract.py", "a", inode=301),
        "executor": artifact("/repo/fan-execute.py", "b", inode=302),
        "input_artifacts": inputs,
        "local_common": artifact("/repo/v25-common.py", "9", inode=307),
        "local_python": artifact("/usr/bin/python3", "c", inode=303),
        "managed_launcher": artifact("/repo/managed.py", "d", inode=304),
        "managed_plan": artifact("/repo/fan-managed.json", "0", inode=305),
        "managed_plan_sha256": "0" * 64,
        "outer_phase_id": OUTER,
        "phase": fan_in.PHASE,
        "producer_argv": [],
        "remote_acquisition_output": "/remote/output/acquisition.json",
        "remote_bundle_root": "/remote/output/bundle",
        "remote_python": artifact("/usr/bin/python3.14", "e", inode=306),
        "remote_runtime_output": "/remote/output/runtime.json",
        "role": fan_in.ROLE,
        "schema": fan_in.PLAN_SCHEMA,
        "source_artifacts": sources,
        "timeout_seconds": 100,
        "v24_phase_id": INNER,
    }
    value["producer_argv"] = fan_in.expected_producer_argv(value)
    return value


def component(value: dict, component_id: str) -> dict:
    return {"component_id": component_id, **copy.deepcopy(value)}


def bound_plans() -> tuple[dict, bytes, dict, bytes]:
    wrapper = wrapper_plan()
    components = []
    artifacts = [
        wrapper["remote_python"],
        *wrapper["source_artifacts"].values(),
        *wrapper["input_artifacts"].values(),
        artifact("/usr/bin/nvidia-smi", "f", inode=500),
    ]
    for index, value in enumerate(artifacts):
        components.append(component(value, f"component_{index:02d}"))
    components.sort(key=lambda value: value["component_id"])
    managed_public = {
        "android": None,
        "bundle_id": "v25_remote_fan_in",
        "components": components,
        "endpoint": "cuda",
        "launcher_component_id": "component_00",
        "mode": "remote_cuda",
        "route": {
            "argv": wrapper["producer_argv"],
            "cwd": "/remote",
            "environment": {
                "CUDA_VISIBLE_DEVICES": GPU,
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "kind": "remote_exec",
            "local_forward": {
                "local_host": "127.0.0.1",
                "local_port": 49121,
                "remote_host": "127.0.0.1",
                "remote_port": 49122,
            },
        },
        "schema": "s39-managed-runtime-launch-plan-v1",
        "ssh": {"gpu_uuid": GPU},
    }
    managed_raw = common.canonical_compact(managed_public)
    managed_sha = hashlib.sha256(managed_raw).hexdigest()
    wrapper["managed_plan_sha256"] = managed_sha
    wrapper["managed_plan"]["bytes"] = len(managed_raw)
    wrapper["managed_plan"]["sha256"] = managed_sha
    wrapper["managed_plan"]["stat"]["size"] = len(managed_raw)
    managed = copy.deepcopy(managed_public)
    managed["_normalized"] = {
        "argv": wrapper["producer_argv"],
        "component_map": {
            row["component_id"]: row
            for row in components
        },
        "environment": copy.deepcopy(managed_public["route"]["environment"]),
        "launcher_path": wrapper["remote_python"]["path"],
    }
    managed["ssh"]["_expected_boot_id"] = RTX_BOOT
    wrapper_raw = common.canonical_bytes(wrapper)
    return wrapper, wrapper_raw, managed, managed_raw


def file_row(path: str, marker: str, size: int = 10) -> dict:
    return {"bytes": size, "path": path, "sha256": marker * 64}


def controller_transport(started: int, completed: int, pid: int) -> dict:
    return {
        "argv": ["/usr/bin/ssh", "host"],
        "clock": "CONTROLLER_MONOTONIC",
        "completed_ns": completed,
        "controller_boot_id": CONTROLLER_BOOT,
        "exit_code": 0,
        "pid": pid,
        "schema": fan_in.CONTROLLER_TRANSPORT_SCHEMA,
        "start_ticks": pid * 10,
        "started_ns": started,
    }


def receipt(
    local_root: str = "/local/bundle",
) -> tuple[dict, dict, bytes, dict, bytes]:
    wrapper, wrapper_raw, managed, managed_raw = bound_plans()
    files = [
        file_row(fan_in.RAW_MANIFEST_NAME, "1"),
        file_row(fan_in.ACQUISITION_NAME, "2"),
        file_row("raw/cuda-monolithic.json", "3"),
        file_row("raw/joint-phone-cuda.json", "4"),
        file_row(fan_in.RUNTIME_IDENTITY_NAME, "5"),
    ]
    files.sort(key=lambda value: value["path"])
    remote_bundle = fan_in.bundle_manifest(
        wrapper["remote_bundle_root"],
        files,
    )
    local_bundle = fan_in.bundle_manifest(local_root, files)
    component_ids = sorted(managed["_normalized"]["component_map"])
    dependencies = sorted(
        [
            {
                **row["stat"],
                "path": row["path"],
                "sha256": row["sha256"],
            }
            for row in managed["components"]
        ],
        key=lambda value: value["path"],
    )
    runtime = {
        "boot_id": RTX_BOOT,
        "bundle_id": managed["bundle_id"],
        "controller_clock": "CONTROLLER_MONOTONIC",
        "controller_observed_ns": 150,
        "endpoint": "cuda",
        "launch_token": "1" * 32,
        "launcher_path": wrapper["remote_python"]["path"],
        "loaded_repo_component_ids": component_ids,
        "pgid": 30,
        "pid": 30,
        "remote_clock": "RTX_CLOCK_MONOTONIC_RAW",
        "remote_observed_ns": 1000,
        "schema": fan_in.RUNTIME_PROCESS_SCHEMA,
        "start_ticks": 40,
        "system_dependencies": dependencies,
    }
    managed_cleanup = {
        "absent": [{"pid": 30, "start_ticks": 40}],
        "boot_id": RTX_BOOT,
        "clock": "RTX_CLOCK_MONOTONIC_RAW",
        "gpu_uuid": GPU,
        "launch_token": "1" * 32,
        "matching_nvml_pids": [],
        "matching_process_groups": [],
        "matching_processes": [],
        "observed_ns": 1300,
        "pgid": 30,
        "pid": 30,
        "schema": fan_in.MANAGED_CLEANUP_SCHEMA,
        "start_ticks": 40,
    }
    source_paths = {
        fan_in.RAW_MANIFEST_NAME: (
            f"{wrapper['remote_bundle_root']}/{fan_in.V24_RAW_MANIFEST_NAME}"
        ),
        fan_in.ACQUISITION_NAME: wrapper["remote_acquisition_output"],
        fan_in.RUNTIME_IDENTITY_NAME: wrapper["remote_runtime_output"],
        "raw/cuda-monolithic.json": (
            f"{wrapper['remote_bundle_root']}/raw/cuda-monolithic.json"
        ),
        "raw/joint-phone-cuda.json": (
            f"{wrapper['remote_bundle_root']}/raw/joint-phone-cuda.json"
        ),
    }
    file_map = {row["path"]: row for row in files}
    snapshots = []
    for index, relative in enumerate(sorted(source_paths), 600):
        row = file_map[relative]
        snapshots.append({
            "artifact": {
                "bytes": row["bytes"],
                "path": source_paths[relative],
                "sha256": row["sha256"],
                "stat": stat_row(row["bytes"], index),
            },
            "materialized_path": relative,
        })
    value = {
        "acquisition_artifact": file_map[fan_in.ACQUISITION_NAME],
        "capture_input_artifacts": {
            "cuda_monolithic": artifact(
                wrapper["capture_input_paths"]["cuda_monolithic"],
                "3",
                inode=701,
            ),
            "joint_phone_cuda": artifact(
                wrapper["capture_input_paths"]["joint_phone_cuda"],
                "4",
                inode=702,
            ),
        },
        "completed_ns": 500,
        "contract_validator": wrapper["contract_validator"],
        "controller_cleanup": {
            "clock": "CONTROLLER_MONOTONIC",
            "execution_transport_absent": True,
            "fetch_transport_absent": True,
            "local_forward_listener_absent": True,
            "managed_transport_absent": True,
            "observed_ns": 490,
        },
        "controller_clock": "CONTROLLER_MONOTONIC",
        "execution_transport": controller_transport(110, 300, 10),
        "executor": wrapper["executor"],
        "fetch_transport": controller_transport(310, 480, 11),
        "gpu_uuid": GPU,
        "local_bundle": local_bundle,
        "local_common": wrapper["local_common"],
        "managed_plan_artifact": wrapper["managed_plan"],
        "managed_plan_sha256": wrapper["managed_plan_sha256"],
        "managed_remote_cleanup": managed_cleanup,
        "managed_transport_process": {
            "argv": ["/usr/bin/ssh", "host"],
            "bundle_id": managed["bundle_id"],
            "endpoint": "cuda",
            "host_boot_id": CONTROLLER_BOOT,
            "managed_launcher_pid": 10,
            "managed_launcher_start_ticks": 100,
            "observed_ns": 160,
            "pid": 12,
            "plan_sha256": wrapper["managed_plan_sha256"],
            "remote_boot_id": RTX_BOOT,
            "schema": fan_in.MANAGED_TRANSPORT_SCHEMA,
            "start_ticks": 120,
        },
        "outer_phase_id": OUTER,
        "phase": fan_in.PHASE,
        "remote_boot_id": RTX_BOOT,
        "remote_bundle": remote_bundle,
        "remote_cleanup": {
            "boot_id": RTX_BOOT,
            "clock": "RTX_CLOCK_MONOTONIC_RAW",
            "gpu_uuid": GPU,
            "nvml_compute_pids": [],
            "observed_ns": 1400,
            "producer_absent": {"pid": 30, "start_ticks": 40},
            "schema": fan_in.REMOTE_CLEANUP_SCHEMA,
        },
        "remote_execution_interval": {
            "clock": "RTX_CLOCK_MONOTONIC_RAW",
            "completed_ns": 1300,
            "phase_closed_ns": 1200,
            "started_ns": 1000,
        },
        "remote_producer_process": runtime,
        "remote_snapshot_artifacts": snapshots,
        "role": fan_in.ROLE,
        "runtime_identity_artifact": file_map[fan_in.RUNTIME_IDENTITY_NAME],
        "schema": fan_in.RECEIPT_SCHEMA,
        "started_ns": 100,
        "system_swap_used_bytes": 4096,
        "v24_phase_id": INNER,
        "wrapper_plan_sha256": hashlib.sha256(wrapper_raw).hexdigest(),
    }
    return value, wrapper, wrapper_raw, managed, managed_raw


def validate(value: dict, bindings: tuple[dict, bytes, dict, bytes]) -> dict:
    return fan_in.validate_receipt(value, *bindings)


class PlanTests(unittest.TestCase):
    def test_plan_passes(self) -> None:
        value, unused_raw, unused_managed, unused_managed_raw = bound_plans()
        self.assertEqual(fan_in.validate_plan(value)["role"], fan_in.ROLE)

    def test_plan_requires_exact_bootstrap(self) -> None:
        value = wrapper_plan()
        value["producer_argv"][3] += "x"
        with self.assertRaises(common.EvidenceError):
            fan_in.validate_plan(value)

    def test_path_alias_fails(self) -> None:
        value = wrapper_plan()
        value["remote_runtime_output"] = "/remote/output/bundle/runtime.json"
        value["producer_argv"] = fan_in.expected_producer_argv(value)
        with self.assertRaises(common.EvidenceError):
            fan_in.validate_plan(value)

    def test_source_set_is_exact(self) -> None:
        value = wrapper_plan()
        del value["source_artifacts"]["v24_common"]
        with self.assertRaises(common.EvidenceError):
            fan_in.validate_plan(value)

    def test_orchestration_plan_is_bound_not_forwarded(self) -> None:
        value = wrapper_plan()
        path = value["input_artifacts"]["orchestration_plan"]["path"]
        self.assertNotIn(path, value["producer_argv"])
        self.assertEqual(fan_in.validate_plan(value), value)


class ReceiptTests(unittest.TestCase):
    def test_receipt_passes(self) -> None:
        value, *bindings = receipt()
        self.assertEqual(validate(value, tuple(bindings))["role"], fan_in.ROLE)

    def test_wrapper_content_splice_fails(self) -> None:
        value, wrapper, wrapper_raw, managed, managed_raw = receipt()
        wrapper = copy.deepcopy(wrapper)
        wrapper["timeout_seconds"] += 1
        with self.assertRaises(common.EvidenceError):
            fan_in.validate_receipt(
                value,
                wrapper,
                wrapper_raw,
                managed,
                managed_raw,
            )

    def test_managed_content_splice_fails(self) -> None:
        value, wrapper, wrapper_raw, managed, managed_raw = receipt()
        managed = copy.deepcopy(managed)
        managed["bundle_id"] = "spliced"
        with self.assertRaises(common.EvidenceError):
            fan_in.validate_receipt(
                value,
                wrapper,
                wrapper_raw,
                managed,
                managed_raw,
            )

    def test_remote_clock_misuse_fails(self) -> None:
        value, *bindings = receipt()
        value["remote_producer_process"]["remote_clock"] = (
            "CONTROLLER_MONOTONIC"
        )
        with self.assertRaises(common.EvidenceError):
            validate(value, tuple(bindings))

    def test_cross_clock_values_are_not_compared(self) -> None:
        value, *bindings = receipt()
        value["remote_producer_process"]["remote_observed_ns"] = 1
        value["remote_execution_interval"]["started_ns"] = 1
        value["remote_execution_interval"]["phase_closed_ns"] = 2
        value["managed_remote_cleanup"]["observed_ns"] = 3
        value["remote_execution_interval"]["completed_ns"] = 3
        value["remote_cleanup"]["observed_ns"] = 4
        validate(value, tuple(bindings))

    def test_child_cleanup_fails(self) -> None:
        value, *bindings = receipt()
        value["managed_remote_cleanup"]["matching_processes"] = [{
            "pid": 31,
        }]
        with self.assertRaises(common.EvidenceError):
            validate(value, tuple(bindings))

    def test_global_nvml_live_fails(self) -> None:
        value, *bindings = receipt()
        value["remote_cleanup"]["nvml_compute_pids"] = [99]
        with self.assertRaises(common.EvidenceError):
            validate(value, tuple(bindings))

    def test_duplicate_snapshot_path_fails(self) -> None:
        value, *bindings = receipt()
        value["remote_snapshot_artifacts"][1]["materialized_path"] = (
            value["remote_snapshot_artifacts"][0]["materialized_path"]
        )
        with self.assertRaises(common.EvidenceError):
            validate(value, tuple(bindings))

    def test_bundle_root_mismatch_fails(self) -> None:
        value, *bindings = receipt()
        value["remote_bundle"]["root_path"] = "/remote/other"
        with self.assertRaises(common.EvidenceError):
            validate(value, tuple(bindings))


class MaterializedBundleTests(unittest.TestCase):
    def build_bundle(self, root: Path):
        cuda_raw = common.canonical_bytes({"schema": "cuda"})
        joint_raw = common.canonical_bytes({"schema": "joint"})
        evidence_raw = common.canonical_bytes({
            "acquisition_id": INNER,
            "event_ns": 110,
            "phase": fan_in.PHASE,
            "phase_id": INNER,
            "role": "quality.corpus",
        })
        raw_manifest = {
            "acquisition_started_ns": 100,
            "artifacts": [{
                "bytes": len(evidence_raw),
                "format": "CANONICAL_ASCII_JSONL",
                "path": "raw/evidence.jsonl",
                "role": "quality.corpus",
                "sha256": common.sha256_bytes(evidence_raw),
            }],
            "candidate_sha256": "1" * 64,
            "clock_id": "HOST_MONOTONIC_RAW",
            "contract_sha256": "2" * 64,
            "phase": fan_in.PHASE,
            "phase_closed_ns": 1200,
            "phase_id": INNER,
            "phase_opened_ns": 90,
            "schema": "s39-cp0-r1-evidence-bundle-v2.1",
        }
        raw_manifest_raw = common.canonical_bytes(raw_manifest)
        runtime = {
            "phase": fan_in.PHASE,
            "phase_id": INNER,
            "schema": "s39-cp0-r1-runtime-identity-v2.4",
        }
        runtime_raw = common.canonical_bytes(runtime)
        acquisition = {
            "artifacts": [
                {
                    "bytes": len(cuda_raw),
                    "path": "raw/cuda-monolithic.json",
                    "role": "capture.cuda_monolithic",
                    "sha256": common.sha256_bytes(cuda_raw),
                },
                {
                    "bytes": len(joint_raw),
                    "path": "raw/joint-phone-cuda.json",
                    "role": "capture.joint_phone_cuda",
                    "sha256": common.sha256_bytes(joint_raw),
                },
            ],
            "completed_ns": 1200,
            "phase": fan_in.PHASE,
            "phase_id": INNER,
            "raw_manifest_name": fan_in.V24_RAW_MANIFEST_NAME,
            "raw_manifest_sha256": common.sha256_bytes(raw_manifest_raw),
            "runtime_identity_sha256": common.sha256_bytes(runtime_raw),
            "schema": "s39-cp0-r1-a-only-acquisition-v2.4",
            "status": "RAW_CAPTURE_COMPLETE_UNEVALUATED",
        }
        values = {
            fan_in.RAW_MANIFEST_NAME: raw_manifest_raw,
            fan_in.ACQUISITION_NAME: common.canonical_bytes(acquisition),
            fan_in.RUNTIME_IDENTITY_NAME: runtime_raw,
            "raw/cuda-monolithic.json": cuda_raw,
            "raw/evidence.jsonl": evidence_raw,
            "raw/joint-phone-cuda.json": joint_raw,
        }
        for relative, raw in values.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        return values

    def bind_bundle(
        self,
        root: Path,
        values: dict[str, bytes],
        capture_values: dict[str, bytes] | None = None,
    ):
        value, wrapper, wrapper_raw, managed, managed_raw = receipt(str(root))
        files = [
            {
                "bytes": len(raw),
                "path": relative,
                "sha256": common.sha256_bytes(raw),
            }
            for relative, raw in sorted(values.items())
        ]
        value["local_bundle"] = fan_in.bundle_manifest(str(root), files)
        value["remote_bundle"] = fan_in.bundle_manifest(
            wrapper["remote_bundle_root"],
            files,
        )
        file_map = {row["path"]: row for row in files}
        value["acquisition_artifact"] = file_map[fan_in.ACQUISITION_NAME]
        value["runtime_identity_artifact"] = file_map[
            fan_in.RUNTIME_IDENTITY_NAME
        ]
        source_paths = {
            fan_in.RAW_MANIFEST_NAME: (
                f"{wrapper['remote_bundle_root']}/"
                f"{fan_in.V24_RAW_MANIFEST_NAME}"
            ),
            fan_in.ACQUISITION_NAME: wrapper["remote_acquisition_output"],
            fan_in.RUNTIME_IDENTITY_NAME: wrapper["remote_runtime_output"],
        }
        value["remote_snapshot_artifacts"] = []
        for index, (relative, raw) in enumerate(sorted(values.items()), 800):
            source = source_paths.get(
                relative,
                f"{wrapper['remote_bundle_root']}/{relative}",
            )
            value["remote_snapshot_artifacts"].append({
                "artifact": {
                    "bytes": len(raw),
                    "path": source,
                    "sha256": common.sha256_bytes(raw),
                    "stat": stat_row(len(raw), index),
                },
                "materialized_path": relative,
            })
        captures = values if capture_values is None else capture_values
        for role, relative in (
            ("cuda_monolithic", "raw/cuda-monolithic.json"),
            ("joint_phone_cuda", "raw/joint-phone-cuda.json"),
        ):
            row = value["capture_input_artifacts"][role]
            row["bytes"] = len(captures[relative])
            row["sha256"] = common.sha256_bytes(captures[relative])
            row["stat"]["size"] = len(captures[relative])
        return value, wrapper, wrapper_raw, managed, managed_raw

    def test_materialized_bundle_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            values = self.build_bundle(root)
            value, wrapper, wrapper_raw, managed, managed_raw = (
                self.bind_bundle(root, values)
            )
            observed = fan_in.validate_materialized_bundle(
                value,
                root,
                wrapper,
                wrapper_raw,
                managed,
                managed_raw,
            )
            self.assertEqual(observed["phase_id"], INNER)

    def test_capture_receipt_for_different_bundle_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            values = self.build_bundle(root)
            capture_values = dict(values)
            capture_values["raw/cuda-monolithic.json"] = (
                common.canonical_bytes({"schema": "different-cuda"})
            )
            value, wrapper, wrapper_raw, managed, managed_raw = (
                self.bind_bundle(root, values, capture_values)
            )
            with self.assertRaisesRegex(
                common.EvidenceError,
                r"bundle\.capture\.cuda_monolithic",
            ):
                fan_in.validate_materialized_bundle(
                    value,
                    root,
                    wrapper,
                    wrapper_raw,
                    managed,
                    managed_raw,
                )

    def test_extra_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.build_bundle(root)
            (root / "extra").write_bytes(b"x")
            value, wrapper, wrapper_raw, managed, managed_raw = receipt(str(root))
            with self.assertRaises(common.EvidenceError):
                fan_in.validate_materialized_bundle(
                    value,
                    root,
                    wrapper,
                    wrapper_raw,
                    managed,
                    managed_raw,
                )


if __name__ == "__main__":
    unittest.main()
