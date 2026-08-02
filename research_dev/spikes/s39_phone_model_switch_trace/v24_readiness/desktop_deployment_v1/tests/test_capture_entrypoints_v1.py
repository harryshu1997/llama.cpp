#!/usr/bin/env python3

from __future__ import annotations

import copy
import contextlib
import datetime
import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parents[1]
V24 = HERE.parent
V24_TESTS = V24 / "tests"
for path in (HERE, V24, V24_TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import artifact_root_capture_v1 as artifact
import fast_fresh_capture_v1 as fresh
from test_v24_readiness import Fixture


common = artifact.capture


def completed(stdout: bytes, returncode: int = 0, stderr: bytes = b""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def remote_json(row: dict, checksum: str | None) -> bytes:
    value = {"stat": row}
    if checksum is not None:
        value["sha256"] = checksum
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )


def android_time(value: int) -> tuple[int, str]:
    seconds, nanoseconds = divmod(value, 1_000_000_000)
    base = datetime.datetime.fromtimestamp(
        seconds,
        tz=datetime.timezone.utc,
    ).strftime("%Y-%m-%d %H:%M:%S")
    return seconds, f"{base}.{nanoseconds:09d} +0000"


def android_stat_line(row: dict) -> str:
    mtime_s, mtime = android_time(row["mtime_ns"])
    ctime_s, ctime = android_time(row["ctime_ns"])
    return (
        f"DEV={row['device_id']}|INO={row['inode']}|SIZE={row['size']}|"
        f"MODE={row['mode']:x}|MTIME_S={mtime_s}|MTIME={mtime}|"
        f"CTIME_S={ctime_s}|CTIME={ctime}"
    )


def android_artifact(row: dict, checksum: str | None) -> bytes:
    line = android_stat_line(row)
    if checksum is None:
        return f"BEFORE\n{line}\n".encode("ascii")
    return (
        f"BEFORE\n{line}\nDIGEST={checksum}\nAFTER\n{line}\n"
    ).encode("ascii")


class FakeRunner:
    def __init__(self):
        self.responses: dict[tuple[str, ...], subprocess.CompletedProcess] = {}
        self.calls: list[list[str]] = []

    def add(self, argv: list[str], stdout: bytes, returncode: int = 0):
        self.responses[tuple(argv)] = completed(stdout, returncode)

    def run(self, argv: list[str], *, timeout: float):
        del timeout
        self.calls.append(argv)
        if tuple(argv) not in self.responses:
            return completed(b"", 97, b"unexpected command")
        return self.responses[tuple(argv)]


class CaptureFixture:
    def __init__(self, root: Path):
        self.base = Fixture(root)
        self.authority = common.load_authority()
        self._bind_capture_sources()
        self.runner = FakeRunner()
        self.rows = {
            row["component_id"]: row
            for row in self.base.artifact_root["components"]
        }
        self.serials = {
            endpoint: self.base.contract["devices"][endpoint]["serial"]
            for endpoint in ("op12", "op15")
        }
        for row in self.rows.values():
            endpoint = row["endpoint"]
            if endpoint == "cuda":
                hash_argv = common.ssh_python_argv(
                    common.CUDA_SSH_TARGET,
                    common.REMOTE_HASH_PYTHON,
                    row["path"],
                )
                stat_argv = common.ssh_python_argv(
                    common.CUDA_SSH_TARGET,
                    common.REMOTE_STAT_PYTHON,
                    row["path"],
                )
                self.runner.add(
                    hash_argv,
                    remote_json(row["stat"], row["sha256"]),
                )
                self.runner.add(stat_argv, remote_json(row["stat"], None))
            else:
                self.runner.add(
                    common.adb_argv(
                        common.PHONE_ADB_PORT,
                        self.serials[endpoint],
                        common.android_stat_command(row["path"], True),
                    ),
                    android_artifact(row["stat"], row["sha256"]),
                )
                self.runner.add(
                    common.adb_argv(
                        common.PHONE_ADB_PORT,
                        self.serials[endpoint],
                        common.android_stat_command(row["path"], False),
                    ),
                    android_artifact(row["stat"], None),
                )
        for inventory in self.base.artifact_root["inventories"]:
            endpoint = inventory["endpoint"]
            if endpoint == "cuda":
                argv = common.ssh_python_argv(
                    common.CUDA_SSH_TARGET,
                    common.REMOTE_INVENTORY_PYTHON,
                    inventory["root"],
                )
                payload = {
                    "bad": [],
                    "files": inventory["paths"],
                }
                self.runner.add(
                    argv,
                    (
                        json.dumps(payload, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    ).encode("ascii"),
                )
            else:
                argv = common.adb_argv(
                    common.PHONE_ADB_PORT,
                    self.serials[endpoint],
                    common._android_inventory_command(inventory["root"]),
                )
                self.runner.add(
                    argv,
                    "".join(f"FILE={path}\n" for path in inventory["paths"]).encode(
                        "ascii"
                    ),
                )
        self._add_device_status()

    def _bind_capture_sources(self):
        sources = {
            "artifact-driver": Path(artifact.__file__).read_bytes(),
            "fresh-driver": Path(fresh.__file__).read_bytes(),
        }
        for row in self.base.plan["components"]:
            if row["component_id"] in sources:
                raw = sources[row["component_id"]]
                row["bytes"] = len(raw)
                row["sha256"] = hashlib.sha256(raw).hexdigest()
        self.base.rewrite("plan", self.base.plan)
        self.base.plan_raw = self.base.plan_path.read_bytes()
        self.base.plan_derived = self.authority.validate_runtime_plan(
            self.base.plan,
            self.base.contract,
            self.base.contract_raw,
            self.base.candidate_raw,
        )
        self.base.artifact_root = self.base._artifact_root()
        self.base.rewrite("artifact_root", self.base.artifact_root)
        self.base.artifact_root_raw = self.base.artifact_root_path.read_bytes()
        self.base.preparation = self.base._preparation()
        self.base.rewrite("preparation", self.base.preparation)
        self.base.preparation_raw = self.base.preparation_path.read_bytes()
        self.base.phase_lock = self.base._phase_lock()
        self.base.rewrite("phase_lock", self.base.phase_lock)
        self.base.phase_lock_raw = self.base.phase_lock_path.read_bytes()
        self.base.fresh = self.base._fresh()
        self.base.rewrite("fresh", self.base.fresh)
        self.base.fresh_raw = self.base.fresh_path.read_bytes()

    def _add_device_status(self):
        devices = self.base._devices()
        cuda = devices["cuda"]
        cuda_stdout = (
            f"HOST={cuda['host']}\n"
            f"BOOT_ID={cuda['host_boot_id']}\n"
            f"SWAP_USED_KB={cuda['system_swap_used_bytes'] // 1024}\n"
            f"GPU_UUID={cuda['gpu_uuid']}\n"
            f"PCI_BUS_ID={cuda['pci_bus_id']}\n"
        ).encode("ascii")
        self.cuda_status_argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            common.CUDA_SSH_TARGET,
            common.CUDA_STATUS,
        ]
        self.runner.add(self.cuda_status_argv, cuda_stdout)
        for endpoint in ("op12", "op15"):
            row = devices[endpoint]
            network = self.base.preparation["devices"][endpoint]
            stdout = (
                f"SERIAL={row['serial']}\n"
                f"BOOT_ID={row['boot_id']}\n"
                f"PRODUCT={row['product']}\n"
                f"MODEL={row['model']}\n"
                f"DEVICE={row['device']}\n"
                f"INTERFACE={network['interface']}\n"
                f"LOCAL_IPV4={network['local_ipv4']}\n"
                f"MEM_AVAILABLE_KB={row['available_bytes'] // 1024}\n"
                f"SWAP_USED_KB={row['system_swap_used_bytes'] // 1024}\n"
                f"THERMAL_STATUS={row['thermal_status']}\n"
            ).encode("ascii")
            argv = common.adb_argv(
                common.PHONE_ADB_PORT,
                self.serials[endpoint],
                common.PHONE_STATUS,
            )
            self.runner.add(argv, stdout)

    def capture_root(self, output: Path | None = None):
        output = output or (self.base.root / "captured-root.json")
        return artifact.capture_artifact_root(
            output=output,
            contract_path=self.base.contract_path,
            candidate_path=self.base.candidate_path,
            runtime_plan_path=self.base.plan_path,
            history_path=self.base.history_path,
            tokenizer_plan_path=self.base.tokenizer_plan_path,
            cuda_ssh_target=common.CUDA_SSH_TARGET,
            phone_adb_port=common.PHONE_ADB_PORT,
            confirmation=common.CONFIRM_ARTIFACT,
            runner=self.runner,
            now_ns=iter((100, 200)).__next__,
        )

    def capture_fresh(self, output: Path | None = None, now=None):
        output = output or (self.base.root / "captured-fresh.json")
        return fresh.capture_fast_fresh(
            output=output,
            phase_id=self.base.phase_lock["phase_id"],
            contract_path=self.base.contract_path,
            artifact_root_path=self.base.artifact_root_path,
            preparation_path=self.base.preparation_path,
            phase_lock_path=self.base.phase_lock_path,
            runtime_plan_path=self.base.plan_path,
            cuda_ssh_target=common.CUDA_SSH_TARGET,
            phone_adb_port=common.PHONE_ADB_PORT,
            confirmation=common.CONFIRM_FRESH,
            runner=self.runner,
            now_ns=now or iter((500, 700)).__next__,
        )


class CaptureEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = CaptureFixture(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_long_capture_is_exact_authority_schema(self):
        value = self.fixture.capture_root()
        self.assertEqual(value, self.fixture.base.artifact_root)
        self.assertEqual(
            (self.fixture.base.root / "captured-root.json").read_bytes(),
            self.fixture.base.artifact_root_raw,
        )

    def test_fast_capture_is_exact_authority_schema(self):
        value = self.fixture.capture_fresh()
        self.assertEqual(value["schema"], fresh.SCHEMA)
        self.assertEqual(
            value["artifact_root_sha256"],
            self.fixture.base.fresh["artifact_root_sha256"],
        )
        self.assertEqual(value["component_stats"], self.fixture.base.fresh["component_stats"])
        self.assertEqual(value["inventories"], self.fixture.base.fresh["inventories"])

    def test_long_capture_rejects_digest_mutation(self):
        row = self.fixture.rows["model.cuda"]
        argv = common.ssh_python_argv(
            common.CUDA_SSH_TARGET,
            common.REMOTE_HASH_PYTHON,
            row["path"],
        )
        self.fixture.runner.add(argv, remote_json(row["stat"], "f" * 64))
        with self.assertRaisesRegex(common.CaptureError, "E_SHA256"):
            self.fixture.capture_root()

    def test_long_capture_rejects_nonregular_component(self):
        row = self.fixture.rows["model.cuda"]
        changed = copy.deepcopy(row["stat"])
        changed["mode"] = stat.S_IFLNK | 0o777
        argv = common.ssh_python_argv(
            common.CUDA_SSH_TARGET,
            common.REMOTE_HASH_PYTHON,
            row["path"],
        )
        self.fixture.runner.add(argv, remote_json(changed, row["sha256"]))
        with self.assertRaisesRegex(common.CaptureError, "E_NOT_REGULAR"):
            self.fixture.capture_root()

    def test_long_capture_rejects_symlink_probe(self):
        row = self.fixture.rows["model.cuda"]
        argv = common.ssh_python_argv(
            common.CUDA_SSH_TARGET,
            common.REMOTE_HASH_PYTHON,
            row["path"],
        )
        self.fixture.runner.add(argv, b"refused\n", returncode=30)
        with self.assertRaisesRegex(common.CaptureError, "E_EXIT"):
            self.fixture.capture_root()

    def test_long_capture_rejects_missing_required_component(self):
        row = self.fixture.rows["model.cuda"]
        argv = common.ssh_python_argv(
            common.CUDA_SSH_TARGET,
            common.REMOTE_HASH_PYTHON,
            row["path"],
        )
        self.fixture.runner.responses.pop(tuple(argv))
        with self.assertRaisesRegex(common.CaptureError, "E_EXIT"):
            self.fixture.capture_root()

    def test_unrelated_bundle_file_is_outside_inventory_claim(self):
        inventory = self.fixture.base.artifact_root["inventories"][0]
        argv = common.ssh_python_argv(
            common.CUDA_SSH_TARGET,
            common.REMOTE_INVENTORY_PYTHON,
            inventory["root"],
        )
        payload = {"bad": [], "files": sorted(inventory["paths"] + ["/extra"])}
        self.fixture.runner.add(
            argv,
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "ascii"
            ),
        )
        value = self.fixture.capture_root()
        self.assertEqual(value["inventories"], self.fixture.base.artifact_root["inventories"])
        self.assertNotIn(argv, self.fixture.runner.calls)

    def test_wrong_adb_port_is_rejected_before_probe(self):
        with self.assertRaisesRegex(common.CaptureError, "phone_adb_port"):
            artifact.capture_artifact_root(
                output=self.fixture.base.root / "bad-port.json",
                contract_path=self.fixture.base.contract_path,
                candidate_path=self.fixture.base.candidate_path,
                runtime_plan_path=self.fixture.base.plan_path,
                history_path=self.fixture.base.history_path,
                tokenizer_plan_path=self.fixture.base.tokenizer_plan_path,
                cuda_ssh_target=common.CUDA_SSH_TARGET,
                phone_adb_port=5037,
                confirmation=common.CONFIRM_ARTIFACT,
                runner=self.fixture.runner,
            )
        self.assertEqual(self.fixture.runner.calls, [])

    def test_wifi_selector_cannot_replace_physical_serial(self):
        self.fixture.serials["op12"] = "172.20.59.72:5555"
        row = self.fixture.rows["model.op12_shard"]
        with self.assertRaisesRegex(common.CaptureError, "physical_serials"):
            common.collect_artifact(
                self.fixture.runner,
                self.fixture.authority,
                "op12",
                row["path"],
                self.fixture.serials,
                include_digest=True,
                timeout=1,
            )

    def test_fast_capture_rejects_post_hash_mutation(self):
        row = self.fixture.rows["model.cuda"]
        changed = copy.deepcopy(row["stat"])
        changed["mtime_ns"] += 1
        argv = common.ssh_python_argv(
            common.CUDA_SSH_TARGET,
            common.REMOTE_STAT_PYTHON,
            row["path"],
        )
        self.fixture.runner.add(argv, remote_json(changed, None))
        with self.assertRaisesRegex(fresh.capture.CaptureError, "E_POST_HASH_MUTATION"):
            self.fixture.capture_fresh()

    def test_fast_capture_rejects_wrong_gpu(self):
        devices = self.fixture.base._devices()
        cuda = devices["cuda"]
        stdout = (
            f"HOST={cuda['host']}\n"
            f"BOOT_ID={cuda['host_boot_id']}\n"
            "SWAP_USED_KB=0\n"
            f"GPU_UUID={'9' * 64}\n"
            f"PCI_BUS_ID={cuda['pci_bus_id']}\n"
        ).encode("ascii")
        self.fixture.runner.add(self.fixture.cuda_status_argv, stdout)
        with self.assertRaisesRegex(fresh.capture.CaptureError, "cuda.uuid"):
            self.fixture.capture_fresh()

    def test_fast_capture_rejects_wrong_boot(self):
        endpoint = "op12"
        row = self.fixture.base._devices()[endpoint]
        network = self.fixture.base.preparation["devices"][endpoint]
        stdout = (
            f"SERIAL={row['serial']}\n"
            "BOOT_ID=aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa\n"
            f"PRODUCT={row['product']}\n"
            f"MODEL={row['model']}\n"
            f"DEVICE={row['device']}\n"
            f"INTERFACE={network['interface']}\n"
            f"LOCAL_IPV4={network['local_ipv4']}\n"
            f"MEM_AVAILABLE_KB={row['available_bytes'] // 1024}\n"
            f"SWAP_USED_KB={row['system_swap_used_bytes'] // 1024}\n"
            f"THERMAL_STATUS={row['thermal_status']}\n"
        ).encode("ascii")
        argv = common.adb_argv(
            common.PHONE_ADB_PORT,
            self.fixture.serials[endpoint],
            common.PHONE_STATUS,
        )
        self.fixture.runner.add(argv, stdout)
        with self.assertRaisesRegex(Exception, "E_FRESH_BOOT"):
            self.fixture.capture_fresh()

    def test_fast_capture_rejects_network_drift(self):
        endpoint = "op15"
        row = self.fixture.base._devices()[endpoint]
        network = self.fixture.base.preparation["devices"][endpoint]
        stdout = (
            f"SERIAL={row['serial']}\n"
            f"BOOT_ID={row['boot_id']}\n"
            f"PRODUCT={row['product']}\n"
            f"MODEL={row['model']}\n"
            f"DEVICE={row['device']}\n"
            f"INTERFACE={network['interface']}\n"
            "LOCAL_IPV4=10.0.0.99\n"
            f"MEM_AVAILABLE_KB={row['available_bytes'] // 1024}\n"
            f"SWAP_USED_KB={row['system_swap_used_bytes'] // 1024}\n"
            f"THERMAL_STATUS={row['thermal_status']}\n"
        ).encode("ascii")
        argv = common.adb_argv(
            common.PHONE_ADB_PORT,
            self.fixture.serials[endpoint],
            common.PHONE_STATUS,
        )
        self.fixture.runner.add(argv, stdout)
        with self.assertRaisesRegex(fresh.capture.CaptureError, "E_FRESH_NETWORK"):
            self.fixture.capture_fresh()

    def test_fast_capture_rejects_slow_interval(self):
        with self.assertRaisesRegex(fresh.capture.CaptureError, "E_FAST_CHECK_SLOW"):
            self.fixture.capture_fresh(
                now=iter((500, 5_000_000_501)).__next__,
            )

    def test_stale_artifact_root_is_rejected_by_phase_lock(self):
        value = copy.deepcopy(self.fixture.base.phase_lock)
        value["event_ns"] = (
            self.fixture.base.artifact_root["completed_ns"]
            + self.fixture.base.contract["gates"]["artifact_root_maximum_age_ns"]
            + 1
        )
        path = self.fixture.base.root / "stale-phase-lock.json"
        path.write_bytes(self.fixture.authority.common.canonical_bytes(value))
        with self.assertRaisesRegex(Exception, "E_ARTIFACT_ROOT_STALE"):
            fresh.capture_fast_fresh(
                output=self.fixture.base.root / "stale-fresh.json",
                phase_id=value["phase_id"],
                contract_path=self.fixture.base.contract_path,
                artifact_root_path=self.fixture.base.artifact_root_path,
                preparation_path=self.fixture.base.preparation_path,
                phase_lock_path=path,
                runtime_plan_path=self.fixture.base.plan_path,
                cuda_ssh_target=common.CUDA_SSH_TARGET,
                phone_adb_port=common.PHONE_ADB_PORT,
                confirmation=common.CONFIRM_FRESH,
                runner=self.fixture.runner,
                now_ns=iter((value["event_ns"] + 1, value["event_ns"] + 2)).__next__,
            )

    def test_preparation_unknown_field_is_rejected(self):
        value = copy.deepcopy(self.fixture.base.preparation)
        value["fabricated"] = True
        path = self.fixture.base.root / "bad-preparation.json"
        path.write_bytes(self.fixture.authority.common.canonical_bytes(value))
        with self.assertRaisesRegex(Exception, "preparation"):
            fresh.capture_fast_fresh(
                output=self.fixture.base.root / "bad-preparation-fresh.json",
                phase_id=self.fixture.base.phase_lock["phase_id"],
                contract_path=self.fixture.base.contract_path,
                artifact_root_path=self.fixture.base.artifact_root_path,
                preparation_path=path,
                phase_lock_path=self.fixture.base.phase_lock_path,
                runtime_plan_path=self.fixture.base.plan_path,
                cuda_ssh_target=common.CUDA_SSH_TARGET,
                phone_adb_port=common.PHONE_ADB_PORT,
                confirmation=common.CONFIRM_FRESH,
                runner=self.fixture.runner,
                now_ns=iter((500, 700)).__next__,
            )

    def test_producers_do_not_load_sibling_authority(self):
        original_artifact = artifact.capture.load_authority
        original_fresh = fresh.capture.load_authority

        def refused():
            raise AssertionError("sibling authority was loaded")

        artifact.capture.load_authority = refused
        fresh.capture.load_authority = refused
        try:
            self.fixture.capture_root(
                self.fixture.base.root / "standalone-root.json"
            )
            self.fixture.capture_fresh(
                self.fixture.base.root / "standalone-fresh.json"
            )
        finally:
            artifact.capture.load_authority = original_artifact
            fresh.capture.load_authority = original_fresh

    def test_artifact_capture_rejects_own_source_mutation(self):
        value = copy.deepcopy(self.fixture.base.plan)
        row = next(
            row for row in value["components"]
            if row["component_id"] == "artifact-driver"
        )
        row["sha256"] = "f" * 64
        path = self.fixture.base.root / "bad-artifact-source-plan.json"
        path.write_bytes(self.fixture.authority.common.canonical_bytes(value))
        with self.assertRaisesRegex(common.CaptureError, "E_CAPTURE_SELF_SHA256"):
            artifact.capture_artifact_root(
                output=self.fixture.base.root / "bad-artifact-source-root.json",
                contract_path=self.fixture.base.contract_path,
                candidate_path=self.fixture.base.candidate_path,
                runtime_plan_path=path,
                history_path=self.fixture.base.history_path,
                tokenizer_plan_path=self.fixture.base.tokenizer_plan_path,
                cuda_ssh_target=common.CUDA_SSH_TARGET,
                phone_adb_port=common.PHONE_ADB_PORT,
                confirmation=common.CONFIRM_ARTIFACT,
                runner=self.fixture.runner,
            )

    def test_artifact_capture_rejects_forged_bundle_digest(self):
        value = copy.deepcopy(self.fixture.base.plan)
        value["cuda_monolithic_launch"]["bundle_sha256"] = "f" * 64
        path = self.fixture.base.root / "bad-bundle-digest-plan.json"
        path.write_bytes(self.fixture.authority.common.canonical_bytes(value))
        with self.assertRaisesRegex(common.CaptureError, "E_CUDA_LAUNCH_BUNDLE_DIGEST"):
            artifact.capture_artifact_root(
                output=self.fixture.base.root / "bad-bundle-digest-root.json",
                contract_path=self.fixture.base.contract_path,
                candidate_path=self.fixture.base.candidate_path,
                runtime_plan_path=path,
                history_path=self.fixture.base.history_path,
                tokenizer_plan_path=self.fixture.base.tokenizer_plan_path,
                cuda_ssh_target=common.CUDA_SSH_TARGET,
                phone_adb_port=common.PHONE_ADB_PORT,
                confirmation=common.CONFIRM_ARTIFACT,
                runner=self.fixture.runner,
            )

    def test_artifact_capture_rejects_joint_nested_mutation(self):
        value = copy.deepcopy(self.fixture.base.plan)
        row = next(
            row for row in value["capture_entrypoints"]
            if row["kind"] == "joint_phone_cuda"
        )
        row["nested_capture_entrypoint_component_ids"] = []
        path = self.fixture.base.root / "bad-joint-nested-plan.json"
        path.write_bytes(self.fixture.authority.common.canonical_bytes(value))
        with self.assertRaisesRegex(common.CaptureError, "joint_phone_cuda.nested"):
            artifact.capture_artifact_root(
                output=self.fixture.base.root / "bad-joint-nested-root.json",
                contract_path=self.fixture.base.contract_path,
                candidate_path=self.fixture.base.candidate_path,
                runtime_plan_path=path,
                history_path=self.fixture.base.history_path,
                tokenizer_plan_path=self.fixture.base.tokenizer_plan_path,
                cuda_ssh_target=common.CUDA_SSH_TARGET,
                phone_adb_port=common.PHONE_ADB_PORT,
                confirmation=common.CONFIRM_ARTIFACT,
                runner=self.fixture.runner,
            )

    def test_fresh_capture_rejects_own_source_mutation(self):
        value = copy.deepcopy(self.fixture.base.plan)
        row = next(
            row for row in value["components"]
            if row["component_id"] == "fresh-driver"
        )
        row["sha256"] = "f" * 64
        path = self.fixture.base.root / "bad-fresh-source-plan.json"
        path.write_bytes(self.fixture.authority.common.canonical_bytes(value))
        with self.assertRaisesRegex(fresh.capture.CaptureError, "E_CAPTURE_SELF_SHA256"):
            fresh.capture_fast_fresh(
                output=self.fixture.base.root / "bad-fresh-source.json",
                phase_id=self.fixture.base.phase_lock["phase_id"],
                contract_path=self.fixture.base.contract_path,
                artifact_root_path=self.fixture.base.artifact_root_path,
                preparation_path=self.fixture.base.preparation_path,
                phase_lock_path=self.fixture.base.phase_lock_path,
                runtime_plan_path=path,
                cuda_ssh_target=common.CUDA_SSH_TARGET,
                phone_adb_port=common.PHONE_ADB_PORT,
                confirmation=common.CONFIRM_FRESH,
                runner=self.fixture.runner,
            )

    def test_fresh_capture_rejects_forged_history_digest_chain(self):
        root = copy.deepcopy(self.fixture.base.artifact_root)
        history = next(
            row for row in root["components"]
            if row["component_id"] == "token_history.mmlu64"
        )
        history["sha256"] = "f" * 64
        root_path = self.fixture.base.root / "forged-root.json"
        root_path.write_bytes(self.fixture.authority.common.canonical_bytes(root))
        root_raw = root_path.read_bytes()

        preparation = copy.deepcopy(self.fixture.base.preparation)
        preparation["artifact_root_sha256"] = hashlib.sha256(root_raw).hexdigest()
        preparation_path = self.fixture.base.root / "forged-preparation.json"
        preparation_path.write_bytes(
            self.fixture.authority.common.canonical_bytes(preparation)
        )
        preparation_raw = preparation_path.read_bytes()

        lock = copy.deepcopy(self.fixture.base.phase_lock)
        lock["artifact_root_sha256"] = hashlib.sha256(root_raw).hexdigest()
        lock["preparation_sha256"] = hashlib.sha256(preparation_raw).hexdigest()
        lock_path = self.fixture.base.root / "forged-lock.json"
        lock_path.write_bytes(self.fixture.authority.common.canonical_bytes(lock))

        with self.assertRaisesRegex(fresh.capture.CaptureError, "E_POST_HASH_DIGEST"):
            fresh.capture_fast_fresh(
                output=self.fixture.base.root / "forged-fresh.json",
                phase_id=lock["phase_id"],
                contract_path=self.fixture.base.contract_path,
                artifact_root_path=root_path,
                preparation_path=preparation_path,
                phase_lock_path=lock_path,
                runtime_plan_path=self.fixture.base.plan_path,
                cuda_ssh_target=common.CUDA_SSH_TARGET,
                phone_adb_port=common.PHONE_ADB_PORT,
                confirmation=common.CONFIRM_FRESH,
                runner=self.fixture.runner,
                now_ns=iter((500, 700)).__next__,
            )

    def test_success_cli_stdout_is_empty(self):
        artifact_args = [
            "--output", "/tmp/root.json",
            "--contract", "/tmp/contract.json",
            "--candidate", "/tmp/candidate.json",
            "--runtime-plan", "/tmp/plan.json",
            "--history", "/tmp/history.json",
            "--tokenizer-plan", "/tmp/tokenizer.json",
            "--cuda-ssh-target", common.CUDA_SSH_TARGET,
            "--phone-adb-port", str(common.PHONE_ADB_PORT),
            "--confirm", common.CONFIRM_ARTIFACT,
        ]
        fresh_args = [
            "--output", "/tmp/fresh.json",
            "--phase-id", "cp0-r1-v24-a-only-test",
            "--contract", "/tmp/contract.json",
            "--root", "/tmp/root.json",
            "--preparation", "/tmp/preparation.json",
            "--phase-lock", "/tmp/lock.json",
            "--runtime-plan", "/tmp/plan.json",
            "--cuda-ssh-target", common.CUDA_SSH_TARGET,
            "--phone-adb-port", str(common.PHONE_ADB_PORT),
            "--confirm", common.CONFIRM_FRESH,
        ]
        output = io.StringIO()
        with mock.patch.object(artifact, "capture_artifact_root", return_value={}):
            with contextlib.redirect_stdout(output):
                self.assertEqual(artifact.main(artifact_args), 0)
        with mock.patch.object(fresh, "capture_fast_fresh", return_value={}):
            with contextlib.redirect_stdout(output):
                self.assertEqual(fresh.main(fresh_args), 0)
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
