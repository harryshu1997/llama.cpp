#!/usr/bin/env python3

import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "managed_runtime_launcher_v1.py"
launcher = types.ModuleType("managed_runtime_launcher_v1")
launcher.__file__ = str(SOURCE)
SOURCE_RAW = SOURCE.read_bytes()
exec(compile(SOURCE_RAW, str(SOURCE), "exec"), launcher.__dict__)


BOOT_ID = "11111111-1111-4111-8111-111111111111"
SERIAL = "3C15AU002CL00000"
SELECTOR = "172.20.173.218:5555"
WORKER = "/data/local/tmp/s39-v23/op15-worker/llama-layersplit"
SHARD = "/data/local/tmp/s39-v23/qwen3-14b-op15.gguf"
ADB = "/usr/bin/adb"
ADB_SHA = "a" * 64
WORKER_SHA = "b" * 64
SHARD_SHA = "c" * 64
GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
STAMP_NS = 1_704_067_200_000_000_000


def stat_row(size):
    return {
        "build_id": None,
        "ctime_ns": STAMP_NS,
        "device_id": 1,
        "inode": 2 + size,
        "mode": stat.S_IFREG | 0o755,
        "mtime_ns": STAMP_NS,
        "size": size,
    }


def local_stat_row(path):
    value = path.stat()
    return {
        "build_id": None,
        "ctime_ns": value.st_ctime_ns,
        "device_id": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "mtime_ns": value.st_mtime_ns,
        "size": value.st_size,
    }


def component(component_id, path, size, checksum):
    return {
        "bytes": size,
        "component_id": component_id,
        "path": path,
        "sha256": checksum,
        "stat": stat_row(size),
    }


def worker_plan():
    return {
        "android": {
            "adb_path": ADB,
            "adb_port": 5038,
            "adb_selector": SELECTOR,
            "adb_sha256": ADB_SHA,
            "boot_id_source": "phase_fresh_snapshot",
            "physical_serial": SERIAL,
            "shutdown_timeout_ms": 1,
            "startup_timeout_ms": 1000,
        },
        "bundle_id": "op15_stagenet",
        "components": [
            component("op15_stagenet.launcher", WORKER, 10, WORKER_SHA),
            component(
                "op15_stagenet.runtime",
                "/data/local/tmp/s39-v23/op15-worker/libllama.so",
                20,
                "d" * 64,
            ),
        ],
        "endpoint": "op15",
        "launcher_component_id": "op15_stagenet.launcher",
        "mode": "android",
        "route": {
            "devices": "HTP0",
            "driver_batch": 64,
            "driver_context": 512,
            "driver_max_prefill": 32,
            "dynamic_cut": True,
            "kind": "stagenet_worker",
            "kv_unified": True,
            "layer_end": 30,
            "layer_start": 0,
            "mode": "stagenet",
            "model_path": SHARD,
            "model_sha256": SHARD_SHA,
            "n_gpu_layers": 99,
            "placement_cert": True,
            "port": 40000,
            "runtime_root": "/data/local/tmp/s39-v23/op15-worker",
        },
        "schema": launcher.PLAN_SCHEMA,
        "ssh": None,
    }


def remote_cuda_plan():
    launcher_path = "/home/zhihao/llama.cpp/build/bin/llama-layersplit"
    nvidia_smi = "/usr/bin/nvidia-smi"
    remote_python = str(Path(sys.executable).resolve())
    return {
        "android": None,
        "bundle_id": "cuda_qwen3_14b",
        "components": [
            component("cuda.launcher", launcher_path, 100, "1" * 64),
            component("cuda.nvidia_smi", nvidia_smi, 200, "2" * 64),
            component("cuda.python", remote_python, 300, "3" * 64),
        ],
        "endpoint": "cuda",
        "launcher_component_id": "cuda.launcher",
        "mode": "remote_cuda",
        "route": {
            "argv": [launcher_path, "--mode", "monov3", "--port", "43000"],
            "cwd": "/home/zhihao/llama.cpp",
            "environment": {
                "CUDA_VISIBLE_DEVICES": GPU_UUID,
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
            },
            "kind": "remote_exec",
            "local_forward": {
                "local_host": "127.0.0.1",
                "local_port": 43100,
                "remote_host": "127.0.0.1",
                "remote_port": 43000,
            },
        },
        "schema": launcher.PLAN_SCHEMA,
        "ssh": {
            "boot_id_source": "phase_fresh_snapshot",
            "connect_timeout_s": 10,
            "gpu_uuid": GPU_UUID,
            "host_key_alias": launcher.REMOTE_CUDA_HOST_KEY_ALIAS,
            "identity_file_path": "/tmp/s39-v25/id_ed25519",
            "identity_file_sha256": "4" * 64,
            "identity_file_stat": stat_row(400),
            "identity_public_key_fingerprint": "SHA256:test",
            "identity_public_key_path": "/tmp/s39-v25/id_ed25519.pub",
            "identity_public_key_sha256": "5" * 64,
            "identity_public_key_stat": stat_row(500),
            "known_hosts_path": "/tmp/s39-v25/known_hosts_v25",
            "known_hosts_sha256": "6" * 64,
            "known_hosts_stat": {
                **stat_row(600),
                "mode": stat.S_IFREG | 0o444,
            },
            "nvidia_smi_path": nvidia_smi,
            "remote_python_path": remote_python,
            "remote_python_sha256": "3" * 64,
            "remote_python_stat": stat_row(300),
            "shutdown_timeout_ms": 1000,
            "ssh_keygen_path": "/usr/bin/ssh-keygen",
            "ssh_keygen_sha256": "7" * 64,
            "ssh_keygen_stat": stat_row(700),
            "ssh_path": "/usr/bin/ssh",
            "ssh_port": 22,
            "ssh_sha256": "8" * 64,
            "ssh_stat": stat_row(800),
            "ssh_target": launcher.REMOTE_CUDA_TARGET,
            "startup_timeout_ms": 30000,
        },
    }


def process_stat(pid, ticks):
    fields = ["S", *["0"] * 18, str(ticks), *["0"] * 3]
    return f"{pid} (llama worker) {' '.join(fields)}\n".encode("ascii")


def process_snapshot(plan, pid=123, ticks=456):
    argv = plan["_normalized"]["argv"]
    cmdline = b"\x00".join(item.encode("ascii") for item in argv) + b"\x00"
    executable = plan["_normalized"]["launcher_path"].encode("ascii")
    return (
        f"SERIAL {SERIAL}\n"
        f"BOOT {BOOT_ID}\n"
        f"EXE {executable.hex()}\n"
        f"CMD {cmdline.hex()}\n"
        f"STAT {process_stat(pid, ticks).hex()}\n"
    ).encode("ascii")


class QueueRunner:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = []

    def run(self, argv, *, timeout, env=None):
        self.calls.append((list(argv), timeout, env))
        if not self.rows:
            raise AssertionError("unexpected subprocess call")
        return self.rows.pop(0)


class PipeProcess:
    def __init__(self, returncode=None, pid=789):
        read_descriptor, write_descriptor = os.pipe()
        self.stdout = os.fdopen(read_descriptor, "rb", buffering=0)
        self.write_descriptor = write_descriptor
        self.pid = pid
        self.returncode = returncode
        self.wait_calls = []

    def poll(self):
        return self.returncode

    def wait(self, *, timeout):
        self.wait_calls.append(timeout)
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = -signal.SIGTERM

    def kill(self):
        self.returncode = -signal.SIGKILL

    def close(self):
        os.close(self.write_descriptor)
        self.stdout.close()


def completed(returncode=0, stdout=b"", stderr=b""):
    return types.SimpleNamespace(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class ManagedRuntimeLauncherTests(unittest.TestCase):
    def validated(self):
        return launcher.validate_plan(copy.deepcopy(worker_plan()))

    def test_worker_command_is_derived_from_typed_route(self):
        value = self.validated()
        normalized = value["_normalized"]
        self.assertEqual(normalized["argv"][0], WORKER)
        self.assertEqual(
            normalized["argv"][1:],
            [
                "-m",
                SHARD,
                "--mode",
                "stagenet",
                "--port",
                "40000",
                "--driver-batch",
                "64",
                "--driver-context",
                "512",
                "--driver-max-prefill",
                "32",
                "--devices",
                "HTP0",
                "-ngl",
                "99",
            ],
        )
        self.assertEqual(normalized["environment"]["LLAMA_LAYER_START"], "0")
        self.assertEqual(normalized["environment"]["LLAMA_LAYER_END"], "30")
        self.assertEqual(normalized["environment"]["LAYERSPLIT_DYNAMIC_CUT"], "1")
        self.assertEqual(normalized["environment"]["LAYERSPLIT_KV_UNIFIED"], "1")

    def test_adb_port_and_wifi_selector_are_fail_closed(self):
        for mutate, message in (
            (lambda value: value["android"].update(adb_port=5037), "adb.port"),
            (
                lambda value: value["android"].update(adb_selector=SERIAL),
                "E_ADB_SELECTOR",
            ),
            (
                lambda value: value["android"].update(adb_selector=""),
                "android.adb_selector",
            ),
        ):
            with self.subTest(message=message):
                value = worker_plan()
                mutate(value)
                with self.assertRaisesRegex(launcher.LaunchError, message):
                    launcher.validate_plan(value)

    def test_process_identity_rejects_serial_boot_pid_ticks_exe_and_argv(self):
        plan = self.validated()
        valid = process_snapshot(plan)
        mutations = {
            "serial": valid.replace(SERIAL.encode(), b"wrong-serial", 1),
            "boot": valid.replace(BOOT_ID.encode(), b"22222222-2222-4222-8222-222222222222", 1),
            "pid": process_snapshot(plan, pid=124),
            "ticks": process_snapshot(plan, ticks=457),
            "exe": valid.replace(WORKER.encode().hex().encode(), b"2f77726f6e67", 1),
        }
        bad_argv = copy.deepcopy(plan)
        bad_argv["_normalized"]["argv"] = [*plan["_normalized"]["argv"], "--extra"]
        mutations["argv"] = process_snapshot(bad_argv)
        for name, raw in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(launcher.LaunchError):
                    launcher.parse_process_snapshot(
                        raw,
                        plan["android"],
                        123,
                        WORKER,
                        plan["_normalized"]["argv"],
                        BOOT_ID,
                        456,
                    )
        ticks, executable = launcher.parse_process_snapshot(
            valid,
            plan["android"],
            123,
            WORKER,
            plan["_normalized"]["argv"],
            BOOT_ID,
            456,
        )
        self.assertEqual((ticks, executable), (456, WORKER))

    def test_remote_process_snapshot_uses_one_quoted_shell_command(self):
        plan = self.validated()
        runner = QueueRunner([completed(stdout=process_snapshot(plan))])
        launcher.remote_process_snapshot(
            runner,
            plan["android"],
            123,
            WORKER,
            plan["_normalized"]["argv"],
            BOOT_ID,
        )
        argv = runner.calls[0][0]
        self.assertEqual(argv[-2], "shell")
        self.assertTrue(argv[-1].startswith("sh -c "))
        self.assertEqual(argv[:5], [ADB, "-P", "5038", "-s", SELECTOR])

    def test_remote_component_is_exact_stat_bound_and_never_rehashes(self):
        plan = self.validated()
        row = plan["components"][0]
        stat_line = (
            "DEV=1|INO=12|SIZE=10|MODE=81ed|"
            "MTIME_S=1704067200|MTIME=2024-01-01 00:00:00.000000000 +0000|"
            "CTIME_S=1704067200|CTIME=2024-01-01 00:00:00.000000000 +0000"
        )
        runner = QueueRunner([completed(stdout=(stat_line + "\n").encode("ascii"))])
        observed = launcher.remote_component(runner, plan["android"], row)
        self.assertEqual(observed["path"], WORKER)
        remote_command = runner.calls[0][0][-1]
        self.assertTrue(remote_command.startswith("sh -c "))
        self.assertNotIn("sha256sum", remote_command)

        changed = stat_line.replace("SIZE=10", "SIZE=11")
        runner = QueueRunner([completed(stdout=(changed + "\n").encode("ascii"))])
        with self.assertRaisesRegex(launcher.LaunchError, r"\.stat"):
            launcher.remote_component(runner, plan["android"], row)

    def test_cleanup_rejects_pid_reuse_and_failed_kill(self):
        plan = self.validated()
        android = plan["android"]
        reused = QueueRunner([
            completed(stdout=(BOOT_ID + "\n").encode("ascii")),
            completed(stdout=process_stat(123, 999)),
        ])
        with self.assertRaisesRegex(launcher.LaunchError, "start_ticks"):
            launcher.cleanup_remote(reused, android, 123, 456, BOOT_ID)

        stale_boot = QueueRunner([
            completed(
                stdout=b"22222222-2222-4222-8222-222222222222\n",
            ),
        ])
        with self.assertRaisesRegex(launcher.LaunchError, "cleanup.boot_id"):
            launcher.cleanup_remote(stale_boot, android, 123, 456, BOOT_ID)
        self.assertEqual(len(stale_boot.calls), 1)

        failed = QueueRunner([
            completed(stdout=(BOOT_ID + "\n").encode("ascii")),
            completed(stdout=process_stat(123, 456)),
            completed(),
            completed(),
            completed(),
            completed(),
        ])
        with self.assertRaisesRegex(launcher.LaunchError, "E_CLEANUP_FAILED"):
            launcher.cleanup_remote(failed, android, 123, 456, BOOT_ID)

    def test_plan_is_inline_canonical_and_digest_bound(self):
        value = worker_plan()
        raw = launcher.canonical_compact(value).decode("ascii")
        checksum = hashlib.sha256(raw.encode("ascii")).hexdigest()
        self.assertEqual(
            launcher.parse_plan_json(raw, checksum)["bundle_id"],
            value["bundle_id"],
        )
        with self.assertRaisesRegex(launcher.LaunchError, "plan.sha256"):
            launcher.parse_plan_json(raw, "0" * 64)
        padded = raw + " "
        with self.assertRaisesRegex(launcher.LaunchError, "plan.canonical"):
            launcher.parse_plan_json(
                padded,
                hashlib.sha256(padded.encode("ascii")).hexdigest(),
            )
        candidate = worker_plan()
        candidate["android"]["boot_id_source"] = "pre_reboot_plan"
        with self.assertRaisesRegex(launcher.LaunchError, "boot_id_source"):
            launcher.validate_plan(candidate)
        candidate = worker_plan()
        candidate["android"]["boot_id"] = BOOT_ID
        with self.assertRaisesRegex(launcher.LaunchError, "E_KEYS"):
            launcher.validate_plan(candidate)
        candidate = worker_plan()
        candidate["components"][0]["stat"]["build_id"] = "not-verified"
        with self.assertRaisesRegex(launcher.LaunchError, "build_id"):
            launcher.validate_plan(candidate)

    def test_legacy_non_remote_plan_normalizes_missing_ssh(self):
        android = worker_plan()
        android.pop("ssh")
        self.assertIsNone(launcher.validate_plan(android)["ssh"])

        local = remote_cuda_plan()
        local["mode"] = "local_cuda"
        local["route"] = {
            key: value
            for key, value in local["route"].items()
            if key != "local_forward"
        }
        local["route"]["kind"] = "local_exec"
        local.pop("ssh")
        self.assertIsNone(launcher.validate_plan(local)["ssh"])

        remote = remote_cuda_plan()
        remote.pop("ssh")
        with self.assertRaisesRegex(launcher.LaunchError, "E_KEYS: plan"):
            launcher.validate_plan(remote)

    def test_silent_pid_marker_times_out_without_blocking(self):
        process = PipeProcess()
        try:
            started = launcher.time.monotonic()
            with self.assertRaisesRegex(launcher.LaunchError, "E_LAUNCH_TIMEOUT"):
                launcher.read_pid_marker(process, started + 0.02)
            self.assertLess(launcher.time.monotonic() - started, 0.5)
        finally:
            process.close()

    def test_pid_marker_preserves_initial_log_bytes(self):
        process = PipeProcess()
        try:
            os.write(process.write_descriptor, b"S39PID 123\nfirst-log\n")
            pid, remainder = launcher.read_pid_marker(
                process,
                launcher.time.monotonic() + 1,
            )
            self.assertEqual(pid, 123)
            self.assertEqual(remainder, b"first-log\n")
        finally:
            process.close()

    def test_forwarding_signal_runs_cleanup_and_preserves_exit_contract(self):
        process = PipeProcess()
        called = []
        try:
            sink = io.BytesIO()
            result = launcher.forward_process_output(
                process,
                b"worker-log\n",
                lambda: int(signal.SIGTERM),
                lambda number: called.append(number),
                0.1,
                sink,
            )
            self.assertEqual(result, 128 + int(signal.SIGTERM))
            self.assertEqual(called, [int(signal.SIGTERM)])
            self.assertEqual(sink.getvalue(), b"worker-log\n")
            self.assertEqual(len(process.wait_calls), 1)
        finally:
            process.close()

    def test_duplicate_process_record_is_rejected_across_chunks(self):
        process = PipeProcess(returncode=0)
        try:
            os.write(process.write_descriptor, b"PROCESS forged\n")
            with self.assertRaisesRegex(
                launcher.LaunchError,
                "E_DUPLICATE_PROCESS_RECORD",
            ):
                launcher.forward_process_output(
                    process,
                    b"RUNTIME",
                    lambda: 0,
                    lambda _number: None,
                    0.1,
                    io.BytesIO(),
                )
        finally:
            process.close()

    def test_remote_cuda_plan_binds_exact_route_gpu_and_forward(self):
        value = launcher.validate_plan(remote_cuda_plan())
        self.assertEqual(value["mode"], "remote_cuda")
        self.assertEqual(
            value["_normalized"]["launcher_path"],
            "/home/zhihao/llama.cpp/build/bin/llama-layersplit",
        )
        probe_argv = launcher.remote_helper_argv(
            value["ssh"],
            "identity",
            {
                "boot_id": BOOT_ID,
                "gpu_uuid": GPU_UUID,
                "nvidia_smi_path": "/usr/bin/nvidia-smi",
            },
        )
        self.assertNotIn("ExitOnForwardFailure=yes", probe_argv)
        launch_argv = launcher.remote_helper_argv(
            value["ssh"],
            "launch",
            {
                "argv": value["_normalized"]["argv"],
                "boot_id": BOOT_ID,
                "cwd": value["_normalized"]["cwd"],
                "environment": value["_normalized"]["environment"],
                "gpu_uuid": GPU_UUID,
                "nvidia_smi_path": "/usr/bin/nvidia-smi",
            },
            value["route"]["local_forward"],
        )
        self.assertIn("ExitOnForwardFailure=yes", launch_argv)
        self.assertIn(
            "127.0.0.1:43100:127.0.0.1:43000",
            launch_argv,
        )
        expected_options = {
            "BatchMode=yes",
            "IdentitiesOnly=yes",
            "StrictHostKeyChecking=yes",
            "HostKeyAlias=172.20.74.85",
            "GlobalKnownHostsFile=/dev/null",
            "PasswordAuthentication=no",
            "KbdInteractiveAuthentication=no",
            "IdentityAgent=none",
            "LogLevel=ERROR",
        }
        self.assertTrue(expected_options.issubset(set(launch_argv)))
        self.assertEqual(launch_argv[-2], launcher.REMOTE_CUDA_TARGET)

        for name, mutate, message in (
            (
                "target",
                lambda item: item["ssh"].update(ssh_target="other@host"),
                "ssh.ssh_target",
            ),
            (
                "gpu",
                lambda item: item["ssh"].update(gpu_uuid="GPU-wrong"),
                "E_GPU_UUID",
            ),
            (
                "environment",
                lambda item: item["route"]["environment"].update(
                    CUDA_VISIBLE_DEVICES="GPU-other-other",
                ),
                "CUDA_VISIBLE_DEVICES",
            ),
            (
                "forward",
                lambda item: item["route"]["local_forward"].update(
                    remote_host="172.20.74.85",
                ),
                "remote_host",
            ),
            (
                "known_hosts",
                lambda item: item["ssh"].update(
                    known_hosts_path="/home/user/.ssh/known_hosts",
                ),
                "E_SHARED_KNOWN_HOSTS",
            ),
        ):
            with self.subTest(name=name):
                candidate = remote_cuda_plan()
                mutate(candidate)
                with self.assertRaisesRegex(launcher.LaunchError, message):
                    launcher.validate_plan(candidate)

    def test_remote_cuda_marker_is_boot_gpu_and_pid_bound(self):
        process = PipeProcess()
        try:
            row = {
                "boot_id": BOOT_ID,
                "gpu_uuid": GPU_UUID,
                "pid": 123,
                "start_ticks": 456,
            }
            os.write(
                process.write_descriptor,
                b"S39CUDA " + launcher.canonical_compact(row) + b"\nlog\n",
            )
            pid, ticks, remote_ns, remainder = launcher.read_remote_cuda_marker(
                process,
                launcher.time.monotonic() + 1,
                BOOT_ID,
                GPU_UUID,
            )
            self.assertEqual(
                (pid, ticks, remote_ns, remainder),
                (123, 456, 0, b"log\n"),
            )
        finally:
            process.close()

        for field, value, message in (
            (
                "boot_id",
                "22222222-2222-4222-8222-222222222222",
                "marker.boot_id",
            ),
            (
                "gpu_uuid",
                "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "marker.gpu_uuid",
            ),
        ):
            process = PipeProcess()
            try:
                row = {
                    "boot_id": BOOT_ID,
                    "gpu_uuid": GPU_UUID,
                    "pid": 123,
                    "start_ticks": 456,
                }
                row[field] = value
                os.write(
                    process.write_descriptor,
                    b"S39CUDA "
                    + launcher.canonical_compact(row)
                    + b"\n",
                )
                with self.assertRaisesRegex(launcher.LaunchError, message):
                    launcher.read_remote_cuda_marker(
                        process,
                        launcher.time.monotonic() + 1,
                        BOOT_ID,
                        GPU_UUID,
                    )
            finally:
                process.close()

    def test_remote_cuda_ssh_files_and_private_key_are_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = root / "id_ed25519"
            subprocess.run(
                [
                    "/usr/bin/ssh-keygen",
                    "-q",
                    "-t",
                    "ed25519",
                    "-N",
                    "",
                    "-f",
                    str(identity),
                ],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            identity.chmod(0o600)
            public = root / "id_ed25519.pub"
            known_hosts = root / "known_hosts_v25"
            known_hosts.write_text(
                "172.20.74.85 ssh-ed25519 AAAATEST\n",
                encoding="ascii",
            )
            known_hosts.chmod(0o444)
            ssh = copy.deepcopy(remote_cuda_plan()["ssh"])
            files = {
                "ssh": Path("/usr/bin/ssh"),
                "known_hosts": known_hosts,
                "identity_file": identity,
                "identity_public_key": public,
                "ssh_keygen": Path("/usr/bin/ssh-keygen"),
            }
            for name, path in files.items():
                path_key = {
                    "ssh": "ssh_path",
                    "known_hosts": "known_hosts_path",
                    "identity_file": "identity_file_path",
                    "identity_public_key": "identity_public_key_path",
                    "ssh_keygen": "ssh_keygen_path",
                }[name]
                ssh[path_key] = str(path)
                ssh[name + "_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                ssh[name + "_stat"] = local_stat_row(path)
            ssh["identity_public_key_fingerprint"] = (
                launcher.public_key_fingerprint(public.read_bytes())
            )
            launcher.verify_ssh(ssh)
            identity.chmod(0o644)
            with self.assertRaises(launcher.LaunchError):
                launcher.verify_ssh(ssh)

    def test_remote_component_is_stat_and_sha256_bound(self):
        value = launcher.validate_plan(remote_cuda_plan())
        component_row = value["components"][0]
        runner = QueueRunner([
            completed(stdout=launcher.canonical_bytes({
                "sha256": component_row["sha256"],
                "stat": component_row["stat"],
            })),
        ])
        with mock.patch.object(launcher, "verify_ssh", return_value=None):
            observed = launcher.remote_cuda_component(
                runner,
                value["ssh"],
                component_row,
            )
        self.assertEqual(observed["path"], component_row["path"])
        self.assertEqual(observed["sha256"], component_row["sha256"])
        command = runner.calls[0][0][-1]
        self.assertNotIn("sha256sum", command)
        encoded_component = shlex.split(command)[-2]
        self.assertEqual(
            json.loads(base64.b64decode(encoded_component, validate=True)),
            component_row,
        )
        self.assertEqual(runner.calls[0][2], launcher.SSH_ENV)

    def test_remote_helper_stat_smoke_needs_no_model(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "small"
            target.write_bytes(b"small")
            interpreter = Path(sys.executable).resolve()
            target_component = {
                "component_id": "test.small",
                "path": str(target),
                "sha256": hashlib.sha256(b"small").hexdigest(),
                "stat": local_stat_row(target),
            }
            payload = base64.b64encode(
                launcher.canonical_compact(target_component)
            ).decode("ascii")
            interpreter_identity = {
                "path": str(interpreter),
                "sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest(),
                "stat": local_stat_row(interpreter),
            }
            encoded_identity = base64.b64encode(
                launcher.canonical_compact(interpreter_identity)
            ).decode("ascii")
            completed = subprocess.run(
                [
                    str(interpreter),
                    "-I",
                    "-c",
                    launcher.REMOTE_CUDA_HELPER,
                    "stat",
                    payload,
                    encoded_identity,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            row = launcher.parse_canonical_json_line(
                completed.stdout,
                "helper.stat",
            )
            self.assertEqual(row["stat"]["size"], 5)
            self.assertEqual(row["sha256"], target_component["sha256"])

            target.write_bytes(b"EVIL!")
            target_component["stat"] = local_stat_row(target)
            completed = subprocess.run(
                [
                    str(interpreter),
                    "-I",
                    "-c",
                    launcher.REMOTE_CUDA_HELPER,
                    "stat",
                    base64.b64encode(
                        launcher.canonical_compact(target_component)
                    ).decode("ascii"),
                    encoded_identity,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(b"component sha256", completed.stderr)

            bad_identity = copy.deepcopy(interpreter_identity)
            bad_identity["stat"]["size"] += 1
            completed = subprocess.run(
                [
                    str(interpreter),
                    "-I",
                    "-c",
                    launcher.REMOTE_CUDA_HELPER,
                    "stat",
                    payload,
                    base64.b64encode(
                        launcher.canonical_compact(bad_identity)
                    ).decode("ascii"),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(b"interpreter stat", completed.stderr)

    def test_remote_python_identity_matches_component_and_command(self):
        plan = remote_cuda_plan()
        value = launcher.validate_plan(copy.deepcopy(plan))
        command = launcher.remote_helper_argv(
            value["ssh"],
            "stat",
            {"path": value["_normalized"]["launcher_path"]},
        )[-1]
        encoded_identity = shlex.split(command)[-1]
        observed = json.loads(
            base64.b64decode(encoded_identity, validate=True).decode("ascii")
        )
        self.assertEqual(
            observed,
            {
                "path": plan["ssh"]["remote_python_path"],
                "sha256": plan["ssh"]["remote_python_sha256"],
                "stat": plan["ssh"]["remote_python_stat"],
            },
        )

        for name, mutate, message in (
            (
                "digest",
                lambda item: item["ssh"].update(
                    remote_python_sha256="9" * 64,
                ),
                "remote_python_sha256",
            ),
            (
                "stat",
                lambda item: item["ssh"]["remote_python_stat"].update(size=301),
                "remote_python_stat",
            ),
            (
                "component",
                lambda item: item["components"][2].update(
                    path="/usr/bin/python-replaced",
                ),
                "E_REMOTE_HELPER_COMPONENT",
            ),
        ):
            with self.subTest(name=name):
                candidate = remote_cuda_plan()
                mutate(candidate)
                with self.assertRaisesRegex(launcher.LaunchError, message):
                    launcher.validate_plan(candidate)

    def test_remote_cuda_launch_emits_one_record_and_uses_forward(self):
        plan = launcher.validate_plan(remote_cuda_plan())
        process = PipeProcess(returncode=0)
        calls = []
        launch_token = "01" * 16

        class PopenRunner:
            def popen(self, argv, *, env=None):
                calls.append((list(argv), env))
                marker = {
                    "boot_id": BOOT_ID,
                    "component_sha256": {
                        component["component_id"]: component["sha256"]
                        for component in plan["components"]
                    },
                    "gpu_uuid": GPU_UUID,
                    "launch_token": launch_token,
                    "pgid": 123,
                    "pid": 123,
                    "remote_observed_ns": 100,
                    "start_ticks": 456,
                }
                os.write(
                    process.write_descriptor,
                    b"S39CUDA "
                    + launcher.canonical_compact(marker)
                    + b"\nworker-log\n",
                )
                return process

        dependencies = {
            component["component_id"]: {
                **component["stat"],
                "path": component["path"],
                "sha256": component["sha256"],
            }
            for component in plan["components"]
        }
        cleanup = []
        sink = io.BytesIO()

        def cleanup_result(*args):
            cleanup.append(args)
            return {
                "absent": [{"pid": 123, "start_ticks": 456}],
                "boot_id": BOOT_ID,
                "clock": "RTX_CLOCK_MONOTONIC_RAW",
                "gpu_uuid": GPU_UUID,
                "launch_token": launch_token,
                "matching_nvml_pids": [],
                "matching_process_groups": [],
                "matching_processes": [],
                "observed_ns": 200,
                "pgid": 123,
                "pid": 123,
                "schema": launcher.REMOTE_CLEANUP_SCHEMA,
                "start_ticks": 456,
            }

        try:
            with (
                mock.patch.object(launcher.os, "urandom", return_value=b"\x01" * 16),
                mock.patch.object(launcher, "verify_ssh", return_value=None),
                mock.patch.object(
                    launcher,
                    "run_remote_helper",
                    return_value={
                        "boot_id": BOOT_ID,
                        "launch_token": launch_token,
                    },
                ),
                mock.patch.object(
                    launcher,
                    "remote_cuda_identity",
                    return_value=None,
                ),
                mock.patch.object(
                    launcher,
                    "remote_cuda_component",
                    side_effect=lambda _runner, _ssh, component: dependencies[
                        component["component_id"]
                    ],
                ),
                mock.patch.object(
                    launcher,
                    "remote_cuda_process_snapshot",
                    return_value=None,
                ),
                mock.patch.object(
                    launcher,
                    "local_process_identity",
                    return_value=987,
                ),
                mock.patch.object(
                    launcher,
                    "local_start_ticks",
                    return_value=654,
                ),
                mock.patch.object(
                    launcher,
                    "cleanup_remote_cuda",
                    side_effect=cleanup_result,
                ),
                mock.patch.object(
                    launcher.sys,
                    "stdout",
                    types.SimpleNamespace(buffer=sink),
                ),
            ):
                result = launcher.launch_remote_cuda(
                    plan,
                    PopenRunner(),
                    BOOT_ID,
                )
            self.assertEqual(result, 0)
            self.assertEqual(len(calls), 1)
            argv, environment = calls[0]
            self.assertEqual(environment, launcher.SSH_ENV)
            self.assertIn("ExitOnForwardFailure=yes", argv)
            self.assertIn(
                "127.0.0.1:43100:127.0.0.1:43000",
                argv,
            )
            output = sink.getvalue()
            self.assertTrue(
                output.startswith(launcher.TRANSPORT_PROCESS_PREFIX)
            )
            self.assertEqual(
                output.count(launcher.TRANSPORT_PROCESS_PREFIX),
                1,
            )
            self.assertEqual(output.count(launcher.PROCESS_PREFIX), 1)
            self.assertTrue(output.endswith(b"\n"))
            (
                transport_line,
                runtime_line,
                worker_log,
                cleanup_line,
            ) = output.splitlines()
            transport = json.loads(
                transport_line[
                    len(launcher.TRANSPORT_PROCESS_PREFIX):
                ].decode("ascii")
            )
            runtime = json.loads(
                runtime_line[len(launcher.PROCESS_PREFIX):].decode("ascii")
            )
            self.assertEqual(
                set(transport),
                {
                    "argv",
                    "bundle_id",
                    "endpoint",
                    "host_boot_id",
                    "managed_launcher_pid",
                    "managed_launcher_start_ticks",
                    "observed_ns",
                    "pid",
                    "plan_sha256",
                    "remote_boot_id",
                    "schema",
                    "start_ticks",
                },
            )
            self.assertEqual(transport["schema"], launcher.TRANSPORT_PROCESS_SCHEMA)
            self.assertEqual(transport["pid"], 789)
            self.assertEqual(transport["start_ticks"], 987)
            self.assertEqual(transport["argv"], argv)
            self.assertEqual(runtime["pid"], 123)
            self.assertEqual(runtime["launch_token"], launch_token)
            self.assertEqual(runtime["remote_observed_ns"], 100)
            self.assertNotEqual(transport["pid"], runtime["pid"])
            self.assertEqual(worker_log, b"worker-log")
            self.assertTrue(cleanup_line.startswith(launcher.REMOTE_CLEANUP_PREFIX))
            self.assertEqual(len(cleanup), 1)
            self.assertEqual(cleanup[0][-3:], (launch_token, 123, 456))
        finally:
            process.close()

    def test_remote_cuda_process_snapshot_is_exact(self):
        plan = launcher.validate_plan(remote_cuda_plan())
        expected = {
            "argv": plan["_normalized"]["argv"],
            "boot_id": BOOT_ID,
            "executable_path": plan["_normalized"]["launcher_path"],
            "gpu_uuid": GPU_UUID,
            "pid": 123,
            "start_ticks": 456,
        }
        with mock.patch.object(
            launcher,
            "run_remote_helper",
            return_value=expected,
        ):
            launcher.remote_cuda_process_snapshot(
                object(),
                plan["ssh"],
                BOOT_ID,
                123,
                456,
                plan["_normalized"]["launcher_path"],
                plan["_normalized"]["argv"],
            )

        for field, bad_value in (
            ("boot_id", "22222222-2222-4222-8222-222222222222"),
            ("gpu_uuid", "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
            ("pid", 124),
            ("start_ticks", 457),
            ("executable_path", "/wrong"),
            ("argv", ["/wrong"]),
        ):
            row = copy.deepcopy(expected)
            row[field] = bad_value
            with self.subTest(field=field), mock.patch.object(
                launcher,
                "run_remote_helper",
                return_value=row,
            ):
                with self.assertRaises(launcher.LaunchError):
                    launcher.remote_cuda_process_snapshot(
                        object(),
                        plan["ssh"],
                        BOOT_ID,
                        123,
                        456,
                        plan["_normalized"]["launcher_path"],
                        plan["_normalized"]["argv"],
                    )

    def test_remote_cuda_pre_marker_timeout_runs_remote_cleanup(self):
        plan = launcher.validate_plan(remote_cuda_plan())
        plan["ssh"]["startup_timeout_ms"] = 1
        process = PipeProcess()
        cleanup = []
        launch_token = "02" * 16
        sink = io.BytesIO()

        class PopenRunner:
            def popen(self, argv, *, env=None):
                del argv, env
                return process

        def cleanup_result(*args):
            cleanup.append(args)
            return {
                "absent": [],
                "boot_id": BOOT_ID,
                "clock": "RTX_CLOCK_MONOTONIC_RAW",
                "gpu_uuid": GPU_UUID,
                "launch_token": launch_token,
                "matching_nvml_pids": [],
                "matching_process_groups": [],
                "matching_processes": [],
                "observed_ns": 200,
                "pgid": 0,
                "pid": 0,
                "schema": launcher.REMOTE_CLEANUP_SCHEMA,
                "start_ticks": 0,
            }

        try:
            with (
                mock.patch.object(launcher.os, "urandom", return_value=b"\x02" * 16),
                mock.patch.object(launcher, "verify_ssh", return_value=None),
                mock.patch.object(launcher, "remote_cuda_identity", return_value=None),
                mock.patch.object(
                    launcher,
                    "remote_cuda_component",
                    side_effect=lambda _runner, _ssh, component: {
                        **component["stat"],
                        "path": component["path"],
                        "sha256": component["sha256"],
                    },
                ),
                mock.patch.object(
                    launcher,
                    "run_remote_helper",
                    return_value={
                        "boot_id": BOOT_ID,
                        "launch_token": launch_token,
                    },
                ),
                mock.patch.object(launcher, "local_process_identity", return_value=987),
                mock.patch.object(launcher, "local_start_ticks", return_value=654),
                mock.patch.object(
                    launcher,
                    "cleanup_remote_cuda",
                    side_effect=cleanup_result,
                ),
                mock.patch.object(
                    launcher.sys,
                    "stdout",
                    types.SimpleNamespace(buffer=sink),
                ),
            ):
                with self.assertRaisesRegex(launcher.LaunchError, "E_LAUNCH_TIMEOUT"):
                    launcher.launch_remote_cuda(plan, PopenRunner(), BOOT_ID)
            self.assertEqual(len(cleanup), 1)
            self.assertEqual(cleanup[0][-3:], (launch_token, 0, 0))
            cleanup_lines = [
                line
                for line in sink.getvalue().splitlines()
                if line.startswith(launcher.REMOTE_CLEANUP_PREFIX)
            ]
            self.assertEqual(len(cleanup_lines), 1)
            cleanup_evidence = json.loads(
                cleanup_lines[0][
                    len(launcher.REMOTE_CLEANUP_PREFIX):
                ].decode("ascii")
            )
            self.assertEqual(cleanup_evidence["launch_token"], launch_token)
            self.assertEqual(cleanup_evidence["matching_nvml_pids"], [])
            self.assertEqual(cleanup_evidence["matching_process_groups"], [])
            self.assertEqual(cleanup_evidence["matching_processes"], [])
        finally:
            process.close()

    def test_local_transport_process_identity_is_live_checked(self):
        executable = str(Path("/usr/bin/sleep").resolve())
        argv = [executable, "10"]
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            ticks = launcher.local_process_identity(
                process.pid,
                executable,
                argv,
                os.getpid(),
            )
            self.assertGreater(ticks, 0)
            with self.assertRaisesRegex(
                launcher.LaunchError,
                "local_process.argv",
            ):
                launcher.local_process_identity(
                    process.pid,
                    executable,
                    [executable, "11"],
                    os.getpid(),
                )
            with self.assertRaisesRegex(
                launcher.LaunchError,
                "local_process.parent_pid",
            ):
                launcher.local_process_identity(
                    process.pid,
                    executable,
                    argv,
                    os.getpid() + 1,
                )
        finally:
            process.terminate()
            process.wait(timeout=5)

    def test_local_adb_hash_is_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adb"
            path.write_bytes(b"adb-client")
            path.chmod(0o755)
            android = copy.deepcopy(worker_plan()["android"])
            android["adb_path"] = str(path)
            android["adb_sha256"] = hashlib.sha256(b"adb-client").hexdigest()
            launcher.verify_adb(android)
            android["adb_sha256"] = "0" * 64
            with self.assertRaisesRegex(launcher.LaunchError, "adb.sha256"):
                launcher.verify_adb(android)

    def test_runtime_record_binds_bundle_components_and_observation(self):
        plan = self.validated()
        dependencies = [
            {**value["stat"], "path": value["path"]}
            for value in reversed(plan["components"])
        ]
        record = launcher.process_record(
            plan,
            123,
            456,
            dependencies,
            789,
            BOOT_ID,
        )
        self.assertEqual(record["schema"], launcher.PROCESS_SCHEMA)
        self.assertEqual(record["bundle_id"], "op15_stagenet")
        self.assertEqual(record["endpoint"], "op15")
        self.assertEqual(record["pid"], 123)
        self.assertEqual(record["start_ticks"], 456)
        self.assertEqual(
            record["loaded_repo_component_ids"],
            ["op15_stagenet.launcher", "op15_stagenet.runtime"],
        )
        self.assertEqual(
            [value["path"] for value in record["system_dependencies"]],
            sorted(value["path"] for value in dependencies),
        )


if __name__ == "__main__":
    unittest.main()
