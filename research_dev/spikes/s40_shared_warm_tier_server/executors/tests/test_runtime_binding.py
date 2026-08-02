#!/usr/bin/env python3

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import socket
import struct
import tempfile
import threading
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
EXECUTORS = HERE.parent
import sys

if str(EXECUTORS) not in sys.path:
    sys.path.insert(0, str(EXECUTORS))

from runtime_binding import (
    ControllerAuthenticator,
    RuntimeBinding,
    RuntimeBindingError,
    await_runtime_binding,
    process_start_time_ticks,
    read_controller_binding_evidence,
)


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def runtime(socket_path: Path, **changes):
    executor = {
        "credits": 1,
        "execute_concurrency": 1,
        "executor_id": "GPU",
        "executor_instance_id": "instance-gpu",
        "expected_peer_pid": os.getpid(),
        "expected_peer_start_time_ticks": process_start_time_ticks(),
        "order": 0,
        "output_limit_bytes": 4096,
        "queue_capacity": 1,
        "role": "GPU",
        "socket_path": str(socket_path),
        "timeout_ms": 1000,
        "transport": "UNIX_SOCKET",
    }
    executor.update(changes.pop("executor", {}))
    value = {
        "c3_profile_lock_sha256": None,
        "configuration": "C1_GPU_ONLY_OPTIMIZED",
        "evidence_root_sha256": "1" * 64,
        "event_log_path": str(socket_path.parent / "events.jsonl"),
        "executors": [executor],
        "initial_models": [],
        "promotion_enabled": True,
        "run_id": "run-1",
        "runtime_plan_sha256": "2" * 64,
        "schema": "llama-server-warm-tier-runtime-v4",
    }
    value.update(changes)
    return value


class RuntimeBindingTests(unittest.TestCase):
    def bind(self, path: Path, socket_path: Path):
        return await_runtime_binding(
            path,
            executor_id="GPU",
            executor_instance_id="instance-gpu",
            run_id="run-1",
            socket_path=socket_path,
            timeout_s=0.03,
        )

    def test_exact_runtime_identity_binds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "runtime.json"
            socket_path = root / "gateway.sock"
            path.write_bytes(canonical_bytes(runtime(socket_path)))
            result = self.bind(path, socket_path)
            self.assertEqual(result.executor_instance_id, "instance-gpu")
            self.assertEqual(result.gateway_pid, os.getpid())
            self.assertEqual(
                result.gateway_start_time_ticks,
                process_start_time_ticks(),
            )
            self.assertEqual(len(result.runtime_config_sha256), 64)

    def test_timeout_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                RuntimeBindingError,
                "wait timed out",
            ):
                self.bind(root / "missing.json", root / "gateway.sock")

    def test_nonfinite_and_oversized_timeout_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for timeout in (float("nan"), float("inf"), 3601):
                with self.subTest(timeout=timeout):
                    with self.assertRaisesRegex(
                        RuntimeBindingError,
                        "startup timeout",
                    ):
                        await_runtime_binding(
                            root / "runtime.json",
                            executor_id="GPU",
                            executor_instance_id="instance-gpu",
                            run_id="run-1",
                            socket_path=root / "gateway.sock",
                            timeout_s=timeout,
                        )

    def test_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            socket_path = root / "gateway.sock"
            target.write_bytes(canonical_bytes(runtime(socket_path)))
            link = root / "runtime.json"
            link.symlink_to(target)
            with self.assertRaises(OSError):
                self.bind(link, socket_path)

    def test_fifo_is_rejected_without_exceeding_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "runtime.json"
            socket_path = root / "gateway.sock"
            os.mkfifo(path)
            started = __import__("time").monotonic()
            with self.assertRaisesRegex(
                RuntimeBindingError,
                "regular file",
            ):
                self.bind(path, socket_path)
            self.assertLess(__import__("time").monotonic() - started, 0.2)

    def test_ancestor_swap_during_open_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            container = Path(temporary)
            benign = container / "benign"
            evil = container / "evil"
            parked = container / "parked"
            benign.mkdir()
            evil.mkdir()
            socket_path = benign / "gateway.sock"
            (benign / "runtime.json").write_bytes(
                canonical_bytes(runtime(socket_path))
            )
            (evil / "runtime.json").write_bytes(
                canonical_bytes(runtime(socket_path))
            )
            real_open = os.open
            swapped = False

            def swapping_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if path == "benign" and not swapped:
                    swapped = True
                    benign.rename(parked)
                    evil.rename(benign)
                    try:
                        return real_open(path, flags, *args, **kwargs)
                    finally:
                        benign.rename(evil)
                        parked.rename(benign)
                return real_open(path, flags, *args, **kwargs)

            with mock.patch("runtime_binding.os.open", swapping_open):
                with self.assertRaisesRegex(
                    RuntimeBindingError,
                    "parent changed while opening|parent was replaced",
                ):
                    self.bind(benign / "runtime.json", socket_path)

    def test_wrong_instance_pid_ticks_socket_and_run_are_rejected(self):
        cases = {
            "instance": {
                "executor": {"executor_instance_id": "other"},
            },
            "PID": {"executor": {"expected_peer_pid": os.getpid() + 1}},
            "start ticks": {
                "executor": {
                    "expected_peer_start_time_ticks":
                        process_start_time_ticks() + 1,
                },
            },
            "socket": {"executor": {"socket_path": "/tmp/other.sock"}},
            "run ID": {"run_id": "other-run"},
        }
        for message, changes in cases.items():
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    path = root / "runtime.json"
                    socket_path = root / "gateway.sock"
                    path.write_bytes(
                        canonical_bytes(runtime(socket_path, **changes))
                    )
                    with self.assertRaisesRegex(
                        RuntimeBindingError,
                        message,
                    ):
                        self.bind(path, socket_path)

    def test_duplicate_noncanonical_and_legacy_config_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "runtime.json"
            socket_path = root / "gateway.sock"
            value = runtime(socket_path)
            raw = canonical_bytes(value)
            path.write_bytes(raw[:-2] + b',\"run_id\":\"run-1\"}\n')
            with self.assertRaisesRegex(
                RuntimeBindingError,
                "duplicate key",
            ):
                self.bind(path, socket_path)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "runtime.json"
            socket_path = root / "gateway.sock"
            value = runtime(socket_path)
            value["schema"] = "llama-server-warm-tier-runtime-v3"
            path.write_bytes(canonical_bytes(value))
            with self.assertRaisesRegex(RuntimeBindingError, "schema"):
                self.bind(path, socket_path)

    def test_path_replacement_during_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "runtime.json"
            socket_path = root / "gateway.sock"
            path.write_bytes(canonical_bytes(runtime(socket_path)))
            replacement = root / "replacement.json"
            replacement.write_bytes(canonical_bytes(runtime(socket_path)))
            real_read = os.read
            replaced = False

            def replacing_read(descriptor, count):
                nonlocal replaced
                block = real_read(descriptor, count)
                if block and not replaced:
                    replaced = True
                    os.replace(replacement, path)
                return block

            with mock.patch("runtime_binding.os.read", replacing_read):
                with self.assertRaisesRegex(
                    RuntimeBindingError,
                    "changed during read|path was replaced",
                ):
                    self.bind(path, socket_path)


class FakeConnection:
    def __init__(self, pid: int, uid: int, gid: int):
        self.credentials = struct.pack("3i", pid, uid, gid)

    def getsockopt(self, level, option, length):
        assert level == socket.SOL_SOCKET
        assert option == socket.SO_PEERCRED
        assert length == struct.calcsize("3i")
        return self.credentials


class ControllerAuthenticatorTests(unittest.TestCase):
    def fixture(self, root: Path, **changes):
        runtime_path = root / "runtime.json"
        runtime_path.write_bytes(b'{"runtime":"fixture"}\n')
        runtime_stat = runtime_path.stat()
        executable_path = Path(os.readlink("/proc/self/exe")).resolve()
        identity_path = root / "controller-identity.json"
        value = {
            "controller_executable_path": str(executable_path),
            "controller_executable_sha256": hashlib.sha256(
                executable_path.read_bytes()
            ).hexdigest(),
            "controller_gid": os.getgid(),
            "controller_pid": os.getpid(),
            "controller_start_time_ticks": process_start_time_ticks(),
            "controller_uid": os.getuid(),
            "host_boot_id": Path(
                "/proc/sys/kernel/random/boot_id"
            ).read_text(encoding="ascii").strip(),
            "run_id": "run-1",
            "runtime_config_device": runtime_stat.st_dev,
            "runtime_config_inode": runtime_stat.st_ino,
            "runtime_config_path": str(runtime_path),
            "runtime_config_sha256": hashlib.sha256(
                runtime_path.read_bytes()
            ).hexdigest(),
            "schema": "s40-controller-identity-lock-v1",
        }
        value.update(changes)
        identity_path.write_bytes(canonical_bytes(value))
        runtime_binding = RuntimeBinding(
            executor_id="GPU",
            executor_instance_id="instance-gpu",
            gateway_pid=os.getpid(),
            gateway_start_time_ticks=process_start_time_ticks(),
            runtime_config_path=str(runtime_path),
            runtime_config_sha256=hashlib.sha256(
                runtime_path.read_bytes()
            ).hexdigest(),
            runtime_config_device=runtime_stat.st_dev,
            runtime_config_inode=runtime_stat.st_ino,
        )
        binding_path = root / "controller-binding.json"
        authenticator = ControllerAuthenticator(
            identity_path,
            binding_path,
            run_id="run-1",
            runtime_binding=runtime_binding,
            timeout_s=0.03,
        )
        connection = FakeConnection(os.getpid(), os.getuid(), os.getgid())
        return (
            authenticator,
            connection,
            identity_path,
            binding_path,
            runtime_path,
            value,
        )

    def test_exact_peer_publishes_binding_before_return(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authenticator, connection, _, binding_path, _, _ = self.fixture(
                root
            )
            result = authenticator.authenticate(connection)
            self.assertTrue(binding_path.is_file())
            record = read_controller_binding_evidence(binding_path)
            self.assertEqual(record["peer_pid"], os.getpid())
            self.assertEqual(
                record["controller_identity_sha256"],
                result.controller_identity_sha256,
            )
            before = binding_path.read_bytes()
            authenticator.authenticate(connection)
            self.assertEqual(binding_path.read_bytes(), before)

    def test_peer_pid_uid_and_gid_substitution_are_rejected(self):
        fields = {
            "PID": (os.getpid() + 1, os.getuid(), os.getgid()),
            "UID": (os.getpid(), os.getuid() + 1, os.getgid()),
            "GID": (os.getpid(), os.getuid(), os.getgid() + 1),
        }
        for message, credentials in fields.items():
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    authenticator, _, _, binding_path, _, _ = self.fixture(
                        root
                    )
                    with self.assertRaisesRegex(
                        RuntimeBindingError,
                        "peer identity",
                    ):
                        authenticator.authenticate(
                            FakeConnection(*credentials)
                        )
                    self.assertFalse(binding_path.exists())

    def test_stale_process_start_and_dead_pid_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authenticator, connection, _, binding_path, _, _ = self.fixture(
                root,
                controller_start_time_ticks=process_start_time_ticks() + 1,
            )
            with self.assertRaisesRegex(
                RuntimeBindingError,
                "start ticks",
            ):
                authenticator.authenticate(connection)
            self.assertFalse(binding_path.exists())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dead_pid = (1 << 30) - 1
            authenticator, _, _, binding_path, _, _ = self.fixture(
                root,
                controller_pid=dead_pid,
            )
            with self.assertRaises(FileNotFoundError):
                authenticator.authenticate(
                    FakeConnection(dead_pid, os.getuid(), os.getgid())
                )
            self.assertFalse(binding_path.exists())

    def test_boot_executable_and_runtime_substitution_are_rejected(self):
        cases = {
            "boot": {"host_boot_id": "wrong-boot"},
            "executable digest": {
                "controller_executable_sha256": "f" * 64,
            },
            "runtime config": {"runtime_config_sha256": "f" * 64},
        }
        for message, changes in cases.items():
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    authenticator, connection, _, binding_path, _, _ = (
                        self.fixture(root, **changes)
                    )
                    with self.assertRaisesRegex(
                        RuntimeBindingError,
                        message,
                    ):
                        authenticator.authenticate(connection)
                    self.assertFalse(binding_path.exists())

    def test_identity_replacement_and_runtime_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                authenticator,
                connection,
                identity_path,
                _,
                runtime_path,
                value,
            ) = self.fixture(root)
            authenticator.authenticate(connection)
            replacement = root / "replacement.json"
            replacement.write_bytes(canonical_bytes(value))
            os.replace(replacement, identity_path)
            with self.assertRaisesRegex(
                RuntimeBindingError,
                "identity lock changed",
            ):
                authenticator.authenticate(connection)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                authenticator,
                connection,
                _,
                _,
                runtime_path,
                _,
            ) = self.fixture(root)
            authenticator.authenticate(connection)
            runtime_path.write_bytes(b'{"runtime":"changed"}\n')
            with self.assertRaisesRegex(
                RuntimeBindingError,
                "runtime config identity",
            ):
                authenticator.authenticate(connection)

    def test_binding_evidence_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authenticator, connection, _, binding_path, _, _ = self.fixture(
                root
            )
            authenticator.authenticate(connection)
            replacement = root / "replacement-binding.json"
            replacement.write_bytes(binding_path.read_bytes())
            os.replace(replacement, binding_path)
            with self.assertRaisesRegex(
                RuntimeBindingError,
                "binding evidence changed",
            ):
                authenticator.authenticate(connection)

    def test_subsequent_connection_uses_executable_identity_not_rehash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authenticator, connection, _, _, _, _ = self.fixture(root)
            authenticator.authenticate(connection)
            with mock.patch(
                "runtime_binding._digest_file_descriptor",
                side_effect=AssertionError("unexpected executable rehash"),
            ):
                authenticator.authenticate(connection)

    def test_executable_identity_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authenticator, connection, _, _, _, _ = self.fixture(root)
            binding = authenticator.authenticate(connection)
            changed = mock.Mock(
                st_ctime_ns=binding.controller_executable_ctime_ns,
                st_dev=binding.controller_executable_device,
                st_ino=binding.controller_executable_inode + 1,
                st_mtime_ns=binding.controller_executable_mtime_ns,
                st_size=binding.controller_executable_size,
            )
            with mock.patch(
                "runtime_binding._file_descriptor_metadata",
                return_value=changed,
            ):
                with self.assertRaisesRegex(
                    RuntimeBindingError,
                    "executable identity changed",
                ):
                    authenticator.authenticate(connection)

    def test_connection_before_lock_times_out_without_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                authenticator,
                connection,
                identity_path,
                binding_path,
                _,
                _,
            ) = self.fixture(root)
            identity_path.unlink()
            with self.assertRaisesRegex(
                RuntimeBindingError,
                "identity wait timed out",
            ):
                authenticator.authenticate(connection)
            self.assertFalse(binding_path.exists())

    def test_first_binding_race_publishes_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authenticator, connection, _, binding_path, _, _ = self.fixture(
                root
            )
            errors = []

            def authenticate():
                try:
                    authenticator.authenticate(connection)
                except BaseException as error:
                    errors.append(error)

            threads = [threading.Thread(target=authenticate) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertTrue(binding_path.is_file())
            read_controller_binding_evidence(binding_path)


if __name__ == "__main__":
    unittest.main()
