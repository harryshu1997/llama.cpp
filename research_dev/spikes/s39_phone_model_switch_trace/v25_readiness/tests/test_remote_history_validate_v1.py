#!/usr/bin/env python3

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import remote_history_validate_v1 as driver


BOOT_ID = "22222222-2222-2222-2222-222222222222"


def stat_row(size: int, mode: int = stat.S_IFREG | 0o555) -> dict:
    return {
        "build_id": None,
        "ctime_ns": 10,
        "device_id": 11,
        "inode": 12,
        "mode": mode,
        "mtime_ns": 13,
        "size": size,
    }


def artifact(path: str, marker: str, size: int = 10) -> dict:
    return {
        "bytes": size,
        "path": path,
        "sha256": marker * 64,
    }


def plan() -> dict:
    inputs = {
        "candidate": artifact(
            "/home/zhihao/s39-v25-a-only/prephase/CP0_R1_CANDIDATE.json",
            "1",
        ),
        "corpus": artifact(
            (
                "/home/zhihao/s39-v25-a-only/prephase/"
                "CP0_R1_MMLU64_CORPUS_V2_2.jsonl"
            ),
            "2",
        ),
        "history": artifact(
            "/home/zhihao/s39-v25-a-only/prephase/token-history.json",
            "3",
        ),
        "tokenizer_plan": artifact(
            "/home/zhihao/s39-v25-a-only/prephase/tokenizer-plan.json",
            "4",
        ),
    }
    support = {
        "history_common": artifact(
            (
                "/home/zhihao/llama.cpp-s40/research_dev/spikes/"
                "s39_phone_model_switch_trace/v24_readiness/history_v1/"
                "history_common_v1.py"
            ),
            "5",
        ),
        "python": artifact("/usr/bin/python3.14", "6", 7_481_192),
        "validator": artifact(
            (
                "/home/zhihao/llama.cpp-s40/research_dev/spikes/"
                "s39_phone_model_switch_trace/v24_readiness/history_v1/"
                "validate_b8_history_v1.py"
            ),
            "7",
        ),
    }
    return {
        "command_argv": [
            support["python"]["path"],
            "-I",
            "-c",
            driver.VALIDATOR_WRAPPER,
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
        ],
        "inputs": inputs,
        "remote_cwd": "/home/zhihao/s39-v25-a-only/prephase",
        "schema": driver.PLAN_SCHEMA,
        "ssh": {
            "boot_id_source": "phase_fresh_snapshot",
            "connect_timeout_s": 10,
            "gpu_uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            "host_key_alias": "172.20.74.85",
            "identity_file_path": "/tmp/id_ed25519",
            "identity_file_sha256": "8" * 64,
            "identity_file_stat": stat_row(100, stat.S_IFREG | 0o600),
            "identity_public_key_fingerprint": "SHA256:test",
            "identity_public_key_path": "/tmp/id_ed25519.pub",
            "identity_public_key_sha256": "9" * 64,
            "identity_public_key_stat": stat_row(101, stat.S_IFREG | 0o644),
            "known_hosts_path": "/tmp/known_hosts_v25",
            "known_hosts_sha256": "a" * 64,
            "known_hosts_stat": stat_row(102, stat.S_IFREG | 0o444),
            "nvidia_smi_path": "/usr/bin/nvidia-smi",
            "remote_python_path": "/usr/bin/python3.14",
            "remote_python_sha256": "6" * 64,
            "remote_python_stat": stat_row(7_481_192),
            "shutdown_timeout_ms": 30_000,
            "ssh_keygen_path": "/usr/bin/ssh-keygen",
            "ssh_keygen_sha256": "b" * 64,
            "ssh_keygen_stat": stat_row(103),
            "ssh_path": "/usr/bin/ssh",
            "ssh_port": 22,
            "ssh_sha256": "c" * 64,
            "ssh_stat": stat_row(104),
            "ssh_target": driver.SSH_TARGET,
            "startup_timeout_ms": 30_000,
        },
        "support": support,
    }


def remote_output(stdout: bytes = b"B8_HISTORY_VALIDATE_PASS\n") -> bytes:
    value = {
        "boot_id": BOOT_ID,
        "observed_inputs": {},
        "observed_support": {},
        "returncode": 0,
        "schema": driver.REMOTE_SCHEMA,
        "stderr_base64": "",
        "stdout_base64": base64.b64encode(stdout).decode("ascii"),
    }
    return base64.b64encode(driver.canonical_bytes(value)) + b"\n"


class PlanTests(unittest.TestCase):
    def test_plan_and_digest_pass(self) -> None:
        value = plan()
        raw = driver.canonical_compact(value)
        parsed = driver.parse_plan(
            raw.decode("ascii"),
            hashlib.sha256(raw).hexdigest(),
        )
        self.assertEqual(parsed, value)

    def test_plan_digest_mismatch_fails(self) -> None:
        value = plan()
        raw = driver.canonical_compact(value)
        with self.assertRaises(driver.DriverError):
            driver.parse_plan(raw.decode("ascii"), "f" * 64)

    def test_history_command_uses_source_bound_wrapper(self) -> None:
        value = driver.validate_plan(plan())
        self.assertEqual(value["command_argv"][3], driver.VALIDATOR_WRAPPER)
        self.assertEqual(value["command_argv"][0], "/usr/bin/python3.14")

    def test_known_hosts_must_be_read_only(self) -> None:
        value = plan()
        value["ssh"]["known_hosts_stat"]["mode"] = stat.S_IFREG | 0o644
        with self.assertRaises(driver.DriverError):
            driver.validate_plan(value)

    def test_ssh_argv_is_hardened_without_post_target_separator(self) -> None:
        argv = driver.ssh_argv(driver.validate_plan(plan()))
        self.assertIn("GlobalKnownHostsFile=/dev/null", argv)
        self.assertIn("PasswordAuthentication=no", argv)
        self.assertIn("KbdInteractiveAuthentication=no", argv)
        self.assertIn("IdentityAgent=none", argv)
        self.assertNotIn("--", argv)
        target_index = argv.index(driver.SSH_TARGET)
        self.assertEqual(target_index, len(argv) - 2)

    def test_empty_validator_stdout_fails(self) -> None:
        with self.assertRaises(driver.DriverError):
            driver.parse_remote(remote_output(b""), BOOT_ID)


class LocalSshIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.private = self.root / "id_ed25519"
        completed = subprocess.run(
            [
                "/usr/bin/ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                "A6000-Server",
                "-f",
                str(self.private),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            self.skipTest("ssh-keygen unavailable")
        self.known_hosts = self.root / "known_hosts_v25"
        self.known_hosts.write_text("example invalid\n", encoding="ascii")
        self.known_hosts.chmod(0o444)

    def tearDown(self) -> None:
        self.known_hosts.chmod(0o600)
        self.temp.cleanup()

    @staticmethod
    def identity(path: Path) -> tuple[str, dict]:
        raw = path.read_bytes()
        metadata = path.stat()
        return hashlib.sha256(raw).hexdigest(), driver.stat_record(metadata)

    def bound_plan(self) -> dict:
        value = plan()
        ssh = value["ssh"]
        public = Path(str(self.private) + ".pub")
        for name, path in (
            ("identity_file", self.private),
            ("identity_public_key", public),
            ("known_hosts", self.known_hosts),
            ("ssh", Path("/usr/bin/ssh")),
            ("ssh_keygen", Path("/usr/bin/ssh-keygen")),
        ):
            digest_value, metadata = self.identity(path)
            ssh[f"{name}_path"] = str(path)
            ssh[f"{name}_sha256"] = digest_value
            ssh[f"{name}_stat"] = metadata
        ssh["identity_public_key_fingerprint"] = driver.public_key_fingerprint(
            public.read_bytes()
        )
        return driver.validate_plan(value)

    def test_private_key_matches_public_key_with_comment(self) -> None:
        driver.verify_ssh(self.bound_plan())

    def test_private_key_mode_fails(self) -> None:
        value = self.bound_plan()
        self.private.chmod(0o644)
        _, value["ssh"]["identity_file_stat"] = self.identity(self.private)
        with self.assertRaises(driver.DriverError):
            driver.verify_ssh(value)

    def test_run_reverifies_local_identity_after_ssh(self) -> None:
        value = self.bound_plan()
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=remote_output(),
            stderr=b"",
        )
        with (
            mock.patch.object(driver, "verify_ssh") as verify,
            mock.patch.object(driver.subprocess, "run", return_value=completed),
        ):
            driver.run(value, BOOT_ID)
        self.assertEqual(verify.call_count, 2)


if __name__ == "__main__":
    unittest.main()
