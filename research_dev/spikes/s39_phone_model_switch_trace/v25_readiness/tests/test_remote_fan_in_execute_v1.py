#!/usr/bin/env python3

from __future__ import annotations

import base64
import copy
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import unittest


HERE = Path(__file__).resolve().parents[1]
TEST_HERE = Path(__file__).resolve().parent


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
contract = load_source(
    "remote_fan_in_contract_v1",
    HERE / "remote_fan_in_contract_v1.py",
    {"v25_common": common},
)
execute = load_source(
    "remote_fan_in_execute_v1",
    HERE / "remote_fan_in_execute_v1.py",
    {
        "remote_fan_in_contract_v1": contract,
        "v25_common": common,
    },
)
execute.common = common
execute.contract = contract
fixtures = load_source(
    "remote_fan_in_contract_fixtures",
    TEST_HERE / "test_remote_fan_in_contract_v1.py",
)


def content_artifact(path: str, raw: bytes, inode: int) -> dict:
    return {
        "bytes": len(raw),
        "content_base64": base64.b64encode(raw).decode("ascii"),
        "path": path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "stat": fixtures.stat_row(len(raw), inode),
    }


def fetch_fixture():
    wrapper, wrapper_raw, managed, managed_raw = fixtures.bound_plans()
    managed["ssh"]["gpu_uuid"] = fixtures.GPU
    runtime = fixtures.receipt()[0]["remote_producer_process"]
    cleanup = fixtures.receipt()[0]["managed_remote_cleanup"]
    cuda_raw = common.canonical_bytes({"schema": "cuda"})
    joint_raw = common.canonical_bytes({"schema": "joint"})
    evidence_raw = common.canonical_bytes({
        "acquisition_id": fixtures.INNER,
        "event_ns": 101,
        "phase": contract.PHASE,
        "phase_id": fixtures.INNER,
        "role": "quality.corpus",
    })
    manifest = {
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
        "phase": contract.PHASE,
        "phase_closed_ns": 1200,
        "phase_id": fixtures.INNER,
        "phase_opened_ns": 90,
        "schema": "s39-cp0-r1-evidence-bundle-v2.1",
    }
    contents = {
        contract.RAW_MANIFEST_NAME: common.canonical_bytes(manifest),
        contract.ACQUISITION_NAME: common.canonical_bytes({"value": "acquisition"}),
        contract.RUNTIME_IDENTITY_NAME: common.canonical_bytes({"value": "runtime"}),
        "raw/cuda-monolithic.json": cuda_raw,
        "raw/evidence.jsonl": evidence_raw,
        "raw/joint-phone-cuda.json": joint_raw,
    }
    sources = {
        contract.RAW_MANIFEST_NAME: (
            f"{wrapper['remote_bundle_root']}/{contract.V24_RAW_MANIFEST_NAME}"
        ),
        contract.ACQUISITION_NAME: wrapper["remote_acquisition_output"],
        contract.RUNTIME_IDENTITY_NAME: wrapper["remote_runtime_output"],
    }
    rows = []
    for index, (relative, raw) in enumerate(sorted(contents.items()), 1000):
        rows.append({
            "artifact": content_artifact(
                sources.get(
                    relative,
                    f"{wrapper['remote_bundle_root']}/{relative}",
                ),
                raw,
                index,
            ),
            "materialized_path": relative,
        })
    value = {
        "capture_input_artifacts": {
            "cuda_monolithic": content_artifact(
                wrapper["capture_input_paths"]["cuda_monolithic"],
                cuda_raw,
                1100,
            ),
            "joint_phone_cuda": content_artifact(
                wrapper["capture_input_paths"]["joint_phone_cuda"],
                joint_raw,
                1101,
            ),
        },
        "files": rows,
        "remote_cleanup": {
            "boot_id": fixtures.RTX_BOOT,
            "clock": "RTX_CLOCK_MONOTONIC_RAW",
            "gpu_uuid": fixtures.GPU,
            "nvml_compute_pids": [],
            "observed_ns": 1400,
            "producer_absent": {
                "pid": runtime["pid"],
                "start_ticks": runtime["start_ticks"],
            },
            "schema": contract.REMOTE_CLEANUP_SCHEMA,
        },
        "schema": execute.FETCH_SCHEMA,
        "system_swap_used_bytes": 0,
    }
    return (
        value,
        wrapper,
        wrapper_raw,
        managed,
        managed_raw,
        runtime,
        cleanup,
        contents,
    )


def validate_fetch(value_and_bindings):
    value, wrapper, _, managed, _, runtime, cleanup, _ = value_and_bindings
    return execute.validate_fetch(
        value,
        wrapper,
        managed,
        runtime,
        cleanup,
    )


def launcher_output(receipt: dict) -> bytes:
    return (
        execute.TRANSPORT_PREFIX
        + common.canonical_bytes(receipt["managed_transport_process"])
        + execute.RUNTIME_PREFIX
        + common.canonical_bytes(receipt["remote_producer_process"])
        + execute.CLEANUP_PREFIX
        + common.canonical_bytes(receipt["managed_remote_cleanup"])
    )


class FakeLauncher:
    def __init__(self, plan: dict):
        self.plan = plan

    def parse_plan_json(self, raw: str, digest: str) -> dict:
        del raw, digest
        return copy.deepcopy(self.plan)


class PlanAndLauncherTests(unittest.TestCase):
    def pinned_plan(self, directory: str):
        root = Path(directory).resolve()
        common_path = root / "v25_common.py"
        contract_path = root / "remote_fan_in_contract_v1.py"
        plan_path = root / "fan-in-plan.json"
        common_raw = (HERE / "v25_common.py").read_bytes()
        contract_raw = (HERE / "remote_fan_in_contract_v1.py").read_bytes()
        common_path.write_bytes(common_raw)
        contract_path.write_bytes(contract_raw)
        wrapper = fixtures.wrapper_plan()
        wrapper["local_common"] = execute.artifact_from_path(
            common_path,
            common_raw,
        )
        wrapper["contract_validator"] = execute.artifact_from_path(
            contract_path,
            contract_raw,
        )
        wrapper_raw = common.canonical_bytes(wrapper)
        plan_path.write_bytes(wrapper_raw)
        return plan_path, wrapper_raw, common_path

    def test_pinned_common_and_contract_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_path, raw, unused_common_path = self.pinned_plan(directory)
            plan, reopened, pinned_common, pinned_contract = (
                execute.load_pinned_plan(
                    plan_path,
                    hashlib.sha256(raw).hexdigest(),
                )
            )
            self.assertEqual(reopened, raw)
            self.assertEqual(plan["role"], contract.ROLE)
            self.assertEqual(
                Path(pinned_common.__file__).name,
                "v25_common.py",
            )
            self.assertEqual(
                Path(pinned_contract.__file__).name,
                "remote_fan_in_contract_v1.py",
            )

    def test_mutated_pinned_common_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_path, raw, common_path = self.pinned_plan(directory)
            common_path.write_bytes(common_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                execute.ExecuteError,
                r"bootstrap\.local_common",
            ):
                execute.load_pinned_plan(
                    plan_path,
                    hashlib.sha256(raw).hexdigest(),
                )

    def test_managed_plan_binds_all_components(self) -> None:
        wrapper, _, managed, managed_raw = fixtures.bound_plans()
        managed["ssh"].update({
            "remote_python_path": wrapper["remote_python"]["path"],
            "remote_python_sha256": wrapper["remote_python"]["sha256"],
            "remote_python_stat": wrapper["remote_python"]["stat"],
            "nvidia_smi_path": "/usr/bin/nvidia-smi",
        })
        observed = execute.validate_managed_plan(
            FakeLauncher(managed),
            managed_raw,
            wrapper["managed_plan_sha256"],
            wrapper,
        )
        self.assertEqual(observed["bundle_id"], managed["bundle_id"])

    def test_unbound_extra_component_fails(self) -> None:
        wrapper, _, managed, managed_raw = fixtures.bound_plans()
        managed["ssh"].update({
            "remote_python_path": wrapper["remote_python"]["path"],
            "remote_python_sha256": wrapper["remote_python"]["sha256"],
            "remote_python_stat": wrapper["remote_python"]["stat"],
            "nvidia_smi_path": "/usr/bin/nvidia-smi",
        })
        managed["components"].append({
            "component_id": "zz_extra",
            **fixtures.artifact("/remote/extra", "9", inode=9999),
        })
        managed["_normalized"]["component_map"]["zz_extra"] = (
            managed["components"][-1]
        )
        with self.assertRaises(execute.ExecuteError):
            execute.validate_managed_plan(
                FakeLauncher(managed),
                managed_raw,
                wrapper["managed_plan_sha256"],
                wrapper,
            )

    def test_cli_is_exact(self) -> None:
        args = execute.parse_args([
            "--plan",
            "/tmp/plan",
            "--plan-sha256",
            "1" * 64,
            "--boot-id",
            fixtures.RTX_BOOT,
            "--bundle-root",
            "/tmp/bundle",
            "--receipt",
            "/tmp/receipt",
            "--execute",
            "--confirm",
            execute.CONFIRMATION,
        ])
        self.assertEqual(args.bundle_root, "/tmp/bundle")


class LauncherEvidenceTests(unittest.TestCase):
    def test_hardened_rows_pass(self) -> None:
        receipt, wrapper, _, managed, _ = fixtures.receipt()
        observed = execute.parse_launcher_output(
            launcher_output(receipt),
            managed,
            wrapper["managed_plan_sha256"],
            fixtures.RTX_BOOT,
        )
        self.assertEqual(observed[1]["launch_token"], "1" * 32)

    def test_missing_cleanup_fails(self) -> None:
        receipt, wrapper, _, managed, _ = fixtures.receipt()
        raw = (
            execute.TRANSPORT_PREFIX
            + common.canonical_bytes(receipt["managed_transport_process"])
            + execute.RUNTIME_PREFIX
            + common.canonical_bytes(receipt["remote_producer_process"])
        )
        with self.assertRaises(execute.ExecuteError):
            execute.parse_launcher_output(
                raw,
                managed,
                wrapper["managed_plan_sha256"],
                fixtures.RTX_BOOT,
            )

    def test_live_child_cleanup_fails(self) -> None:
        receipt, wrapper, _, managed, _ = fixtures.receipt()
        receipt["managed_remote_cleanup"]["matching_processes"] = [{
            "pgid": 30,
            "pid": 31,
            "start_ticks": 41,
        }]
        with self.assertRaises(common.EvidenceError):
            execute.parse_launcher_output(
                launcher_output(receipt),
                managed,
                wrapper["managed_plan_sha256"],
                fixtures.RTX_BOOT,
            )


class FetchValidationTests(unittest.TestCase):
    def test_fetch_passes(self) -> None:
        result = validate_fetch(fetch_fixture())
        self.assertEqual(
            result[1]["root_path"],
            "/remote/output/bundle",
        )

    def test_stale_boot_fails(self) -> None:
        values = fetch_fixture()
        values[0]["remote_cleanup"]["boot_id"] = fixtures.CONTROLLER_BOOT
        with self.assertRaises(execute.ExecuteError):
            validate_fetch(values)

    def test_global_nvml_live_fails(self) -> None:
        values = fetch_fixture()
        values[0]["remote_cleanup"]["nvml_compute_pids"] = [123]
        with self.assertRaises(execute.ExecuteError):
            validate_fetch(values)

    def test_missing_file_fails(self) -> None:
        values = fetch_fixture()
        values[0]["files"] = [
            row
            for row in values[0]["files"]
            if row["materialized_path"] != "raw/evidence.jsonl"
        ]
        with self.assertRaises(execute.ExecuteError):
            validate_fetch(values)

    def test_extra_file_fails(self) -> None:
        values = fetch_fixture()
        values[0]["files"].append({
            "artifact": content_artifact(
                "/remote/output/bundle/raw/extra",
                b"x",
                1200,
            ),
            "materialized_path": "raw/extra",
        })
        values[0]["files"].sort(key=lambda row: row["materialized_path"])
        with self.assertRaises(execute.ExecuteError):
            validate_fetch(values)

    def test_truncated_file_fails(self) -> None:
        values = fetch_fixture()
        artifact = values[0]["files"][0]["artifact"]
        raw = base64.b64decode(artifact["content_base64"])
        artifact["content_base64"] = base64.b64encode(raw[:-1]).decode("ascii")
        with self.assertRaises(execute.ExecuteError):
            validate_fetch(values)

    def test_fetch_mutation_fails(self) -> None:
        values = fetch_fixture()
        capture = values[0]["capture_input_artifacts"]["cuda_monolithic"]
        replacement = common.canonical_bytes({"schema": "mutated"})
        capture.update({
            "bytes": len(replacement),
            "content_base64": base64.b64encode(replacement).decode("ascii"),
            "sha256": hashlib.sha256(replacement).hexdigest(),
        })
        capture["stat"]["size"] = len(replacement)
        with self.assertRaises(execute.ExecuteError):
            validate_fetch(values)

    def test_duplicate_logical_path_fails(self) -> None:
        values = fetch_fixture()
        values[0]["files"][1]["materialized_path"] = (
            values[0]["files"][0]["materialized_path"]
        )
        values[0]["files"].sort(key=lambda row: row["materialized_path"])
        with self.assertRaises(execute.ExecuteError):
            validate_fetch(values)

    def test_bundle_root_source_mismatch_fails(self) -> None:
        values = fetch_fixture()
        row = next(
            row
            for row in values[0]["files"]
            if row["materialized_path"] == "raw/evidence.jsonl"
        )
        row["artifact"]["path"] = "/remote/other/raw/evidence.jsonl"
        with self.assertRaises(execute.ExecuteError):
            validate_fetch(values)


class MaterializationTests(unittest.TestCase):
    def test_materializes_and_reopens(self) -> None:
        values = fetch_fixture()
        contents = values[-1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "bundle"
            manifest = execute.materialize_bundle(root, contents)
            self.assertEqual(manifest["file_count"], len(contents))
            self.assertEqual(
                execute.scan_bundle(root),
                manifest["files"],
            )

    def test_partial_local_output_fails(self) -> None:
        values = fetch_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "bundle"
            root.mkdir()
            (root / "partial").write_bytes(b"x")
            with self.assertRaises(execute.ExecuteError):
                execute.materialize_bundle(root, values[-1])

    def test_timeout_kills_process_group(self) -> None:
        with self.assertRaisesRegex(execute.ExecuteError, "E_TEST_TIMEOUT"):
            execute.run_controller_process(
                ["/bin/sh", "-c", "sleep 30 & wait"],
                1,
                1024,
                "TEST",
            )


class RemoteHelperTests(unittest.TestCase):
    def local_artifact(self, path: Path) -> dict:
        raw = path.read_bytes()
        metadata = path.stat()
        return {
            "bytes": len(raw),
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "stat": {
                "build_id": None,
                "ctime_ns": metadata.st_ctime_ns,
                "device_id": metadata.st_dev,
                "inode": metadata.st_ino,
                "mode": metadata.st_mode,
                "mtime_ns": metadata.st_mtime_ns,
                "size": metadata.st_size,
            },
        }

    def test_remote_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            bundle = root / "bundle"
            raw_dir = bundle / "raw"
            raw_dir.mkdir(parents=True)
            target = root / "target"
            target.write_bytes(b"x")
            (raw_dir / "linked").symlink_to(target)
            capture_cuda = root / "cuda.json"
            capture_joint = root / "joint.json"
            runtime = root / "runtime.json"
            acquisition = root / "acquisition.json"
            for path in (capture_cuda, capture_joint, runtime, acquisition):
                path.write_bytes(b"{}\n")
            nvidia = root / "nvidia-smi"
            nvidia.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *--query-gpu=uuid*) printf '%s\\n' \"$2\" ;;\n"
                "  *--query-compute-apps=pid*) : ;;\n"
                "  *) exit 1 ;;\n"
                "esac\n",
                encoding="ascii",
            )
            nvidia.chmod(0o755)
            python_path = Path(os.path.realpath(sys.executable))
            payload = {
                "boot_id": Path(
                    "/proc/sys/kernel/random/boot_id"
                ).read_text(encoding="ascii").strip(),
                "capture_input_paths": {
                    "cuda_monolithic": str(capture_cuda),
                    "joint_phone_cuda": str(capture_joint),
                },
                "gpu_uuid": fixtures.GPU,
                "nvidia_smi": self.local_artifact(nvidia),
                "producer_pid": 2_000_000_000,
                "producer_start_ticks": 1,
                "remote_acquisition_output": str(acquisition),
                "remote_bundle_root": str(bundle),
                "remote_python": self.local_artifact(python_path),
                "remote_runtime_output": str(runtime),
            }
            encoded = base64.b64encode(
                common.canonical_compact(payload)
            ).decode("ascii")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    execute.REMOTE_FETCH_SOURCE,
                    encoded,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"E_SYMLINK", completed.stderr)


if __name__ == "__main__":
    unittest.main()
