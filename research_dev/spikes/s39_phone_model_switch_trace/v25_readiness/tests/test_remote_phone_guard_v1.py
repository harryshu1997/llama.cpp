#!/usr/bin/env python3

from __future__ import annotations

import copy
import base64
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "remote_phone_guard_v1.py"
SPEC = importlib.util.spec_from_file_location("remote_phone_guard_v1", SOURCE)
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


RTX_BOOT = "11111111-2222-3333-4444-555555555555"
OP12_BOOT = "22222222-3333-4444-5555-666666666666"
OP15_BOOT = "33333333-4444-5555-6666-777777777777"
CONTROLLER_BOOT = "44444444-5555-6666-7777-888888888888"


def stat_row(size: int = 100, mode: int = stat.S_IFREG | 0o755) -> dict:
    return {
        "build_id": None,
        "ctime_ns": 10,
        "device_id": 11,
        "inode": 12,
        "mode": mode,
        "mtime_ns": 13,
        "size": size,
    }


def artifact(path: str, marker: str, size: int = 100) -> dict:
    return {
        "bytes": size,
        "path": path,
        "sha256": marker * 64,
        "stat": stat_row(size),
    }


def content_artifact(path: str, raw: bytes) -> dict:
    value = artifact(path, "0", len(raw))
    value["sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def artifact_sets() -> tuple[dict, dict]:
    local = {
        "adb": artifact("/opt/s39/local/adb", "1", 101),
        "helper": artifact(
            "/opt/s39/local/remote_phone_guard_v1.py",
            "2",
            102,
        ),
        "python": artifact("/opt/s39/local/python3", "3", 103),
    }
    remote = {
        "adb": artifact("/opt/s39/remote/adb", "4", 104),
        "helper": artifact(
            "/opt/s39/remote/remote_phone_guard_v1.py",
            "2",
            102,
        ),
        "python": artifact("/opt/s39/remote/python3", "5", 105),
    }
    return local, remote


def phone_plan(
    phone: str,
    boot_id: str,
    ipv4: str,
    interface: str,
    port: int,
) -> dict:
    return {
        "boot_id": boot_id,
        "forbidden_listen_ports": [port],
        "forbidden_processes": [
            {
                "executable_path": f"/data/local/tmp/s39/{phone}-worker",
                "sha256": ("6" if phone == "op12" else "7") * 64,
            },
        ],
        "interface": interface,
        "physical_serial": guard.PHONE_SERIALS[phone],
        "wifi_ipv4": ipv4,
        "wifi_selector": f"{ipv4}:5555",
    }


def plan() -> dict:
    local, remote = artifact_sets()
    value = {
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
            "boot_id": RTX_BOOT,
            "executable_path": remote["adb"]["path"],
            "listen_host": "127.0.0.1",
            "listen_port": 5038,
            "pid": 88,
            "start_ticks": 99,
        },
        "desktop_forbidden_listen_ports": [39312, 39315],
        "desktop_forbidden_processes": [
            {
                "executable_path": "/opt/s39/remote/op12-worker",
                "sha256": "a" * 64,
            },
            {
                "executable_path": "/opt/s39/remote/op15-worker",
                "sha256": "b" * 64,
            },
        ],
        "forbid_adb_forward_for_selectors": True,
        "gpu_uuid": guard.GPU_UUID,
        "inner_phase_id": "cp0-r1-v24-a-only-test",
        "local_artifacts": local,
        "local_policy_artifact": artifact(
            "/opt/s39/local/guard-policy.json",
            "0",
        ),
        "outer_phase_id": "cp0-r1-v25-a-only-test",
        "phase": "A_ONLY",
        "phones": {
            "op12": phone_plan(
                "op12",
                OP12_BOOT,
                "172.20.59.72",
                "wlan0",
                39312,
            ),
            "op15": phone_plan(
                "op15",
                OP15_BOOT,
                "172.20.173.218",
                "wlan1",
                39315,
            ),
        },
        "remote_artifacts": remote,
        "remote_policy": {},
        "remote_policy_artifact": artifact(
            "/opt/s39/remote/guard-policy.json",
            "0",
        ),
        "rtx_boot_id": RTX_BOOT,
        "schema": guard.PLAN_SCHEMA,
        "ssh_transport": {
            "argv_by_moment": {
                "after": [],
                "before": [],
            },
            "connect_timeout_seconds": 10,
            "host_key_alias": "172.20.74.85",
            "identity_file": artifact(
                "/opt/s39/local/id_ed25519",
                "6",
                106,
            ),
            "known_hosts": artifact(
                "/opt/s39/local/known_hosts",
                "7",
                107,
            ),
            "remote_python": remote["python"],
            "ssh": artifact("/usr/bin/ssh", "8", 108),
            "ssh_port": 22,
            "ssh_target": guard.SSH_TARGET,
        },
        "timeout_seconds": 60,
    }
    value["remote_policy"] = guard.expected_remote_policy(value)
    policy_raw = guard.canonical_bytes(value["remote_policy"])
    value["local_policy_artifact"] = content_artifact(
        "/opt/s39/local/guard-policy.json",
        policy_raw,
    )
    value["remote_policy_artifact"] = content_artifact(
        "/opt/s39/remote/guard-policy.json",
        policy_raw,
    )
    value["ssh_transport"]["identity_file"]["stat"]["mode"] = (
        stat.S_IFREG | 0o600
    )
    value["ssh_transport"]["known_hosts"]["stat"]["mode"] = (
        stat.S_IFREG | 0o444
    )
    for moment in ("before", "after"):
        value["ssh_transport"]["argv_by_moment"][moment] = (
            guard.expected_ssh_argv(value, moment)
        )
    return value


def phone_receipt(phone: str, moment: str) -> dict:
    source = plan()["phones"][phone]
    marker = {
        ("before", "op12"): "8",
        ("before", "op15"): "9",
        ("after", "op12"): "a",
        ("after", "op15"): "b",
    }[(moment, phone)]
    return {
        "adb_forwards": [],
        "adb_state": "device",
        "boot_id": source["boot_id"],
        "interface": source["interface"],
        "matching_listeners": [],
        "matching_processes": [],
        "physical_serial": source["physical_serial"],
        "raw_snapshot_artifact": artifact(
            f"/evidence/{moment}-{phone}-snapshot.json",
            marker,
            200,
        ),
        "wifi_ipv4": source["wifi_ipv4"],
        "wifi_selector": source["wifi_selector"],
    }


def receipt(moment: str) -> dict:
    is_before = moment == "before"
    started = 10 if is_before else 30
    completed = 20 if is_before else 40
    controller_observed = 1_000_000 if is_before else 2_000_000
    pid = 101 if is_before else 102
    ticks = 1001 if is_before else 1002
    plan_sha256 = "c" * 64
    return {
        "adb_server_process": {
            **plan()["adb_server_process"],
            "executable_sha256": plan()["remote_artifacts"]["adb"][
                "sha256"
            ],
            "listener_inode": 123,
            "observed_ns": 15 if is_before else 35,
        },
        "clock_name": guard.RTX_CLOCK_NAME,
        "completed_ns": completed,
        "desktop_matching_listeners": [],
        "desktop_matching_processes": [],
        "desktop_raw_snapshot_artifact": artifact(
            f"/evidence/{moment}-desktop-snapshot.json",
            "d" if is_before else "e",
            210,
        ),
        "gpu_uuid": guard.GPU_UUID,
        "inner_phase_id": "cp0-r1-v24-a-only-test",
        "moment": moment,
        "outer_phase_id": "cp0-r1-v25-a-only-test",
        "phase": "A_ONLY",
        "phones": {
            "op12": phone_receipt("op12", moment),
            "op15": phone_receipt("op15", moment),
        },
        "plan_sha256": plan_sha256,
        "remote_output_artifact": artifact(
            f"/evidence/{moment}-remote-output.json",
            "f" if is_before else "0",
            220,
        ),
        "rtx_boot_id": RTX_BOOT,
        "schema": guard.RECEIPT_SCHEMA,
        "ssh_transport_cleanup": {
            "clock_name": guard.CONTROLLER_CLOCK_NAME,
            "controller_boot_id": CONTROLLER_BOOT,
            "observed_ns": controller_observed + 10,
            "pid": pid,
            "process_absent": True,
            "schema": guard.SSH_CLEANUP_SCHEMA,
            "start_ticks": ticks,
        },
        "ssh_transport_process": {
            "argv": plan()["ssh_transport"]["argv_by_moment"][moment],
            "clock_name": guard.CONTROLLER_CLOCK_NAME,
            "controller_boot_id": CONTROLLER_BOOT,
            "observed_ns": controller_observed,
            "pid": pid,
            "plan_sha256": plan_sha256,
            "remote_boot_id": RTX_BOOT,
            "schema": guard.SSH_PROCESS_SCHEMA,
            "start_ticks": ticks,
        },
        "started_ns": started,
    }


def make_file(path: Path, raw: bytes, mode: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)
    return guard.artifact_from_path(path, str(path))


def executable_plan(root: Path) -> dict:
    local_root = root / "local"
    remote_root = root / "remote"
    helper_raw = SOURCE.read_bytes()
    local = {
        "adb": make_file(local_root / "adb", b"local-adb\n", 0o755),
        "helper": make_file(
            local_root / "remote_phone_guard_v1.py",
            helper_raw,
            0o644,
        ),
        "python": make_file(
            local_root / "python3",
            b"local-python\n",
            0o755,
        ),
    }
    remote = {
        "adb": make_file(remote_root / "adb", b"remote-adb\n", 0o755),
        "helper": make_file(
            remote_root / "remote_phone_guard_v1.py",
            helper_raw,
            0o644,
        ),
        "python": make_file(
            remote_root / "python3",
            b"remote-python\n",
            0o755,
        ),
    }
    value = {
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
            "boot_id": RTX_BOOT,
            "executable_path": remote["adb"]["path"],
            "listen_host": "127.0.0.1",
            "listen_port": 5038,
            "pid": 88,
            "start_ticks": 99,
        },
        "desktop_forbidden_listen_ports": [39312, 39315],
        "desktop_forbidden_processes": [
            {
                "executable_path": str(remote_root / "op12-worker"),
                "sha256": "a" * 64,
            },
            {
                "executable_path": str(remote_root / "op15-worker"),
                "sha256": "b" * 64,
            },
        ],
        "forbid_adb_forward_for_selectors": True,
        "gpu_uuid": guard.GPU_UUID,
        "inner_phase_id": "cp0-r1-v24-a-only-executor",
        "local_artifacts": local,
        "local_policy_artifact": {},
        "outer_phase_id": "cp0-r1-v25-a-only-executor",
        "phase": "A_ONLY",
        "phones": {
            "op12": phone_plan(
                "op12",
                OP12_BOOT,
                "172.20.59.72",
                "wlan0",
                39312,
            ),
            "op15": phone_plan(
                "op15",
                OP15_BOOT,
                "172.20.173.218",
                "wlan1",
                39315,
            ),
        },
        "remote_artifacts": remote,
        "remote_policy": {},
        "remote_policy_artifact": {},
        "rtx_boot_id": RTX_BOOT,
        "schema": guard.PLAN_SCHEMA,
        "ssh_transport": {
            "argv_by_moment": {"after": [], "before": []},
            "connect_timeout_seconds": 10,
            "host_key_alias": "172.20.74.85",
            "identity_file": make_file(
                local_root / "id_ed25519",
                b"test-key\n",
                0o600,
            ),
            "known_hosts": make_file(
                local_root / "known_hosts",
                b"test-host-key\n",
                0o444,
            ),
            "remote_python": remote["python"],
            "ssh": make_file(local_root / "ssh", b"test-ssh\n", 0o755),
            "ssh_port": 22,
            "ssh_target": guard.SSH_TARGET,
        },
        "timeout_seconds": 60,
    }
    value["remote_policy"] = guard.expected_remote_policy(value)
    policy_raw = guard.canonical_bytes(value["remote_policy"])
    value["local_policy_artifact"] = make_file(
        local_root / "guard-policy.json",
        policy_raw,
        0o600,
    )
    value["remote_policy_artifact"] = make_file(
        remote_root / "guard-policy.json",
        policy_raw,
        0o600,
    )
    for moment in ("before", "after"):
        value["ssh_transport"]["argv_by_moment"][moment] = (
            guard.expected_ssh_argv(value, moment)
        )
    guard.validate_plan(value)
    return value


class Counter:
    def __init__(self, value: int = 1000) -> None:
        self.value = value

    def __call__(self) -> int:
        self.value += 10
        return self.value


class FakeCommandRunner:
    def __init__(
        self,
        value: dict,
        *,
        forwards: bytes = b"",
        serial_override: str | None = None,
        process_output: bytes = b"OK\n",
        process_returncode: int = 0,
    ) -> None:
        self.value = value
        self.forwards = forwards
        self.serial_override = serial_override
        self.process_output = process_output
        self.process_returncode = process_returncode
        self.calls = []

    def __call__(self, argv: list[str], unused_timeout: int) -> dict:
        del unused_timeout
        self.calls.append(list(argv))
        stdout = b""
        returncode = 0
        if argv[-2:] == ["forward", "--list"]:
            stdout = self.forwards
        else:
            selector = argv[argv.index("-s") + 1]
            phone = next(
                name
                for name in ("op12", "op15")
                if self.value["phones"][name]["wifi_selector"] == selector
            )
            expected = self.value["phones"][phone]
            if argv[-1:] == ["get-state"]:
                stdout = b"device\n"
            elif argv[-3:] == [
                "shell",
                "cat",
                "/proc/sys/kernel/random/boot_id",
            ]:
                stdout = (expected["boot_id"] + "\n").encode("ascii")
            elif argv[-3:] == ["shell", "getprop", "ro.serialno"]:
                serial = self.serial_override or expected["physical_serial"]
                stdout = (serial + "\n").encode("ascii")
            elif "addr" in argv:
                stdout = (
                    f"1: {expected['interface']} inet "
                    f"{expected['wifi_ipv4']}/24 scope global "
                    f"{expected['interface']}\n"
                ).encode("ascii")
            elif guard.PHONE_PROC_SCAN_SCRIPT in argv:
                stdout = self.process_output
                returncode = self.process_returncode
            elif argv[-4:] == [
                "shell",
                "cat",
                "/proc/net/tcp",
                "/proc/net/tcp6",
            ]:
                stdout = (
                    b"sl local_address rem_address st tx_queue rx_queue "
                    b"tr tm->when retrnsmt uid timeout inode\n"
                )
            else:
                raise AssertionError(argv)
        return {
            "argv": argv,
            "returncode": returncode,
            "stderr": b"",
            "stdout": stdout,
        }


def fake_adb_observer(clock: Counter):
    def observe(adb_artifact, expected, unused_clock) -> dict:
        del unused_clock
        return {
            **expected,
            "executable_sha256": adb_artifact["sha256"],
            "listener_inode": 1234,
            "observed_ns": clock(),
        }

    return observe


def execute_remote_fixture(
    value: dict,
    root: Path,
    *,
    runner=None,
    process_scanner=lambda unused: [],
    listener_scanner=lambda unused: [],
    boot_id: str = RTX_BOOT,
) -> tuple[dict, bytes]:
    clock = Counter()
    if runner is None:
        runner = FakeCommandRunner(value["remote_policy"])
    return guard.execute_remote(
        value["remote_policy"],
        value["remote_policy_artifact"]["sha256"],
        "before",
        command_runner=runner,
        clock=clock,
        boot_reader=lambda: boot_id,
        gpu_reader=lambda: [guard.GPU_UUID],
        process_scanner=process_scanner,
        listener_scanner=listener_scanner,
        adb_server_observer=fake_adb_observer(clock),
        output_root=root,
    )


class FakePopen:
    def __init__(
        self,
        argv,
        *,
        stdout: bytes,
        timeout: bool = False,
        **unused,
    ) -> None:
        del unused
        self.argv = argv
        self.pid = 4321
        self.returncode = 0
        self._stdout = stdout
        self._timeout = timeout

    def communicate(self, timeout: int):
        if self._timeout:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return self._stdout, b""

    def poll(self):
        return self.returncode

    def wait(self, timeout: int):
        del timeout
        return self.returncode


def execute_controller_fixture(
    value: dict,
    root: Path,
    remote_raw: bytes,
    *,
    popen_factory=None,
    absent_checker=lambda unused_pid, unused_ticks: True,
    group_absent_checker=lambda unused_pgid: True,
    terminator=lambda unused_process: None,
) -> dict:
    packet = base64.b64encode(remote_raw) + b"\n"
    if popen_factory is None:
        popen_factory = lambda argv, **unused: FakePopen(
            argv,
            stdout=packet,
        )
    plan_sha256 = hashlib.sha256(guard.canonical_bytes(value)).hexdigest()
    return guard.execute_controller(
        value,
        plan_sha256,
        "before",
        root / "receipt.json",
        popen_factory=popen_factory,
        clock=Counter(100_000),
        boot_reader=lambda: CONTROLLER_BOOT,
        ticks_reader=lambda unused_pid: 4567,
        process_executable_reader=lambda unused_pid, unused_ticks: Path(
            value["ssh_transport"]["ssh"]["path"]
        ).read_bytes(),
        absent_checker=absent_checker,
        group_absent_checker=group_absent_checker,
        terminator=terminator,
        executable_path=value["local_artifacts"]["python"]["path"],
    )


class PlanTests(unittest.TestCase):
    def reject(self, mutate) -> None:
        value = copy.deepcopy(plan())
        mutate(value)
        with self.assertRaises(guard.GuardError):
            guard.validate_plan(value)

    def test_valid_plan(self) -> None:
        value = plan()
        self.assertIs(guard.validate_plan(value), value)

    def test_client_start_adb_argv_is_not_a_daemon_identity(self) -> None:
        self.reject(
            lambda value: value["adb_server_process"].update(
                argv=[
                    value["remote_artifacts"]["adb"]["path"],
                    "-P",
                    "5038",
                    "nodaemon",
                    "server",
                ]
            )
        )

    def test_outer_schema_is_exact(self) -> None:
        mutations = [
            lambda value: value.update(extra=True),
            lambda value: value.update(schema="wrong"),
            lambda value: value.update(phase="B_ONLY"),
            lambda value: value.update(
                outer_phase_id="cp0-r1-v24-a-only-test"
            ),
            lambda value: value.update(
                inner_phase_id="cp0-r1-v24-a-only-other"
            ),
            lambda value: value.update(rtx_boot_id=OP12_BOOT),
            lambda value: value.update(gpu_uuid="GPU-" + "0" * 36),
            lambda value: value.update(adb_server_port=5037),
            lambda value: value.update(
                forbid_adb_forward_for_selectors=False
            ),
            lambda value: value.update(timeout_seconds=True),
            lambda value: value.update(timeout_seconds=601),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.reject(mutate)

    def test_artifacts_are_exact_and_content_bound(self) -> None:
        mutations = [
            lambda value: value["local_artifacts"].update(extra={}),
            lambda value: value["local_artifacts"]["python"].update(bytes=True),
            lambda value: value["local_artifacts"]["python"]["stat"].update(
                size=104
            ),
            lambda value: value["local_artifacts"]["adb"]["stat"].update(
                mode=stat.S_IFREG | 0o644
            ),
            lambda value: value["remote_artifacts"]["helper"].update(
                sha256="d" * 64
            ),
            lambda value: value["remote_artifacts"]["helper"].update(
                path="../helper.py"
            ),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.reject(mutate)

    def test_ssh_transport_is_exact(self) -> None:
        mutations = [
            lambda value: value["ssh_transport"].update(
                ssh_target="other@172.20.74.85"
            ),
            lambda value: value["ssh_transport"]["identity_file"]["stat"].update(
                mode=stat.S_IFREG | 0o644
            ),
            lambda value: value["ssh_transport"]["known_hosts"]["stat"].update(
                mode=stat.S_IFREG | 0o644
            ),
            lambda value: value["ssh_transport"]["argv_by_moment"][
                "before"
            ].append("--forged"),
            lambda value: value["ssh_transport"].update(
                remote_python=artifact("/usr/bin/python3.13", "9", 109)
            ),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.reject(mutate)

    def test_remote_policy_is_exact_and_non_self_referential(self) -> None:
        mutations = [
            lambda value: value["remote_policy"].update(
                rtx_boot_id=OP12_BOOT
            ),
            lambda value: value["remote_policy_artifact"].update(
                sha256="f" * 64
            ),
            lambda value: value["local_policy_artifact"].update(
                path=value["remote_policy_artifact"]["path"]
            ),
            lambda value: value["desktop_forbidden_processes"][0].update(
                sha256="f" * 64
            ),
            lambda value: value["desktop_forbidden_listen_ports"].append(
                39312
            ),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.reject(mutate)

    def test_phone_identity_and_policy_are_exact(self) -> None:
        def alias_selector(value) -> None:
            value["phones"]["op15"]["wifi_ipv4"] = "172.20.59.72"
            value["phones"]["op15"]["wifi_selector"] = "172.20.59.72:5555"

        def unsorted_processes(value) -> None:
            rows = value["phones"]["op12"]["forbidden_processes"]
            rows.extend([
                {
                    "executable_path": "/data/local/tmp/s39/a-worker",
                    "sha256": "e" * 64,
                },
            ])

        mutations = [
            lambda value: value["phones"].update(op16={}),
            lambda value: value["phones"]["op12"].update(
                physical_serial="wrong"
            ),
            lambda value: value["phones"]["op12"].update(
                boot_id=value["rtx_boot_id"]
            ),
            lambda value: value["phones"]["op12"].update(
                wifi_ipv4="127.0.0.1",
                wifi_selector="127.0.0.1:5555",
            ),
            lambda value: value["phones"]["op12"].update(
                wifi_selector="172.20.59.72:5556"
            ),
            lambda value: value["phones"]["op12"].update(interface="wlan 0"),
            lambda value: value["phones"]["op12"].update(
                forbidden_processes=[]
            ),
            lambda value: value["phones"]["op12"][
                "forbidden_processes"
            ][0].update(executable_path="relative"),
            lambda value: value["phones"]["op12"][
                "forbidden_processes"
            ][0].update(sha256="A" * 64),
            unsorted_processes,
            lambda value: value["phones"]["op12"].update(
                forbidden_listen_ports=[]
            ),
            lambda value: value["phones"]["op12"].update(
                forbidden_listen_ports=[39315, 39312]
            ),
            lambda value: value["phones"]["op12"].update(
                forbidden_listen_ports=[True]
            ),
            alias_selector,
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.reject(mutate)


class ParsePlanTests(unittest.TestCase):
    def test_canonical_newline_plan_and_digest_pass(self) -> None:
        raw = guard.canonical_bytes(plan())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_bytes(raw)
            parsed, parsed_raw = guard.parse_plan(
                path,
                hashlib.sha256(raw).hexdigest(),
            )
        self.assertEqual(parsed, plan())
        self.assertEqual(parsed_raw, raw)

    def test_digest_mismatch_fails(self) -> None:
        raw = guard.canonical_bytes(plan())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_bytes(raw)
            with self.assertRaises(guard.GuardError):
                guard.parse_plan(path, "f" * 64)

    def test_missing_newline_fails(self) -> None:
        raw = guard.canonical_bytes(plan())[:-1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_bytes(raw)
            with self.assertRaises(guard.GuardError):
                guard.parse_plan(path, hashlib.sha256(raw).hexdigest())

    def test_duplicate_key_fails(self) -> None:
        raw = b'{"schema":"a","schema":"b"}\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_bytes(raw)
            with self.assertRaises(guard.GuardError):
                guard.parse_plan(path, hashlib.sha256(raw).hexdigest())


class ReceiptTests(unittest.TestCase):
    def reject(self, mutate, moment: str = "before") -> None:
        value = copy.deepcopy(receipt(moment))
        mutate(value)
        with self.assertRaises(guard.GuardError):
            guard.validate_receipt(value)

    def test_valid_receipts(self) -> None:
        for moment in ("before", "after"):
            with self.subTest(moment=moment):
                value = receipt(moment)
                self.assertIs(guard.validate_receipt(value), value)

    def test_receipt_identity_and_interval_are_exact(self) -> None:
        mutations = [
            lambda value: value.update(extra=True),
            lambda value: value.update(schema="wrong"),
            lambda value: value.update(phase="PAIR"),
            lambda value: value.update(
                inner_phase_id="cp0-r1-v24-a-only-other"
            ),
            lambda value: value.update(plan_sha256="A" * 64),
            lambda value: value.update(rtx_boot_id=OP12_BOOT),
            lambda value: value.update(gpu_uuid="wrong"),
            lambda value: value.update(clock_name="CLOCK_MONOTONIC"),
            lambda value: value.update(moment="during"),
            lambda value: value.update(started_ns=True),
            lambda value: value.update(completed_ns=value["started_ns"]),
            lambda value: value.update(
                desktop_matching_processes=[{"pid": 1}]
            ),
            lambda value: value.update(
                desktop_matching_listeners=[{"port": 1}]
            ),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.reject(mutate)

    def test_phone_absence_rows_are_exact(self) -> None:
        mutations = [
            lambda value: value["phones"]["op12"].update(extra=True),
            lambda value: value["phones"]["op12"].update(adb_state="offline"),
            lambda value: value["phones"]["op12"].update(
                physical_serial="wrong"
            ),
            lambda value: value["phones"]["op12"].update(
                wifi_selector="172.20.59.72:5556"
            ),
            lambda value: value["phones"]["op12"].update(
                matching_processes=[{"pid": 1}]
            ),
            lambda value: value["phones"]["op12"].update(
                matching_listeners=[{"port": 39312}]
            ),
            lambda value: value["phones"]["op12"].update(
                adb_forwards=[{"local": "tcp:1"}]
            ),
            lambda value: value["phones"]["op12"][
                "raw_snapshot_artifact"
            ]["stat"].update(size=201),
            lambda value: value["phones"]["op15"][
                "raw_snapshot_artifact"
            ].update(
                path=value["phones"]["op12"][
                    "raw_snapshot_artifact"
                ]["path"]
            ),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.reject(mutate)

    def test_ssh_process_and_cleanup_are_exact(self) -> None:
        mutations = [
            lambda value: value["ssh_transport_process"].update(extra=True),
            lambda value: value["ssh_transport_process"].update(
                schema="wrong"
            ),
            lambda value: value["ssh_transport_process"].update(
                clock_name="CLOCK_MONOTONIC"
            ),
            lambda value: value["ssh_transport_process"].update(
                controller_boot_id=RTX_BOOT
            ),
            lambda value: value["ssh_transport_process"].update(
                remote_boot_id=OP12_BOOT
            ),
            lambda value: value["ssh_transport_process"].update(
                plan_sha256="d" * 64
            ),
            lambda value: value["ssh_transport_process"].update(
                argv=["ssh", "target"]
            ),
            lambda value: value["ssh_transport_process"].update(pid=True),
            lambda value: value["ssh_transport_cleanup"].update(
                process_absent=False
            ),
            lambda value: value["ssh_transport_cleanup"].update(pid=999),
            lambda value: value["ssh_transport_cleanup"].update(
                observed_ns=1
            ),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.reject(mutate)


class PairTests(unittest.TestCase):
    def reject(self, mutate) -> None:
        before = receipt("before")
        after = receipt("after")
        mutate(before, after)
        with self.assertRaises(guard.GuardError):
            guard.validate_pair(before, after)

    def test_valid_pair(self) -> None:
        before = receipt("before")
        after = receipt("after")
        self.assertEqual(guard.validate_pair(before, after), (before, after))

    def test_pair_linkage_is_exact(self) -> None:
        mutations = [
            lambda before, after: (
                before.update(moment="after"),
                after.update(moment="before"),
            ),
            lambda before, after: after.update(plan_sha256="d" * 64),
            lambda before, after: after.update(
                outer_phase_id="cp0-r1-v25-a-only-other",
                inner_phase_id="cp0-r1-v24-a-only-other",
            ),
            lambda before, after: after.update(rtx_boot_id=(
                "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            )),
            lambda before, after: after.update(started_ns=19),
            lambda before, after: after["phones"]["op12"].update(
                interface="wlan2"
            ),
            lambda before, after: after["ssh_transport_process"].update(
                controller_boot_id=(
                    "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                )
            ),
            lambda before, after: after["ssh_transport_cleanup"].update(
                controller_boot_id=(
                    "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                )
            ),
            lambda before, after: after["phones"]["op12"][
                "raw_snapshot_artifact"
            ].update(
                path=before["phones"]["op12"][
                    "raw_snapshot_artifact"
                ]["path"]
            ),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.reject(mutate)

    def test_controller_and_rtx_clocks_are_not_compared(self) -> None:
        before = receipt("before")
        after = receipt("after")
        before["ssh_transport_process"]["observed_ns"] = 9_000_000
        before["ssh_transport_cleanup"]["observed_ns"] = 9_000_001
        after["ssh_transport_process"]["observed_ns"] = 1
        after["ssh_transport_cleanup"]["observed_ns"] = 2
        guard.validate_pair(before, after)


class ExecutorTests(unittest.TestCase):
    def test_cli_modes_are_exact_and_confirmation_gated(self) -> None:
        remote = guard.parse_args([
            "--remote",
            "--policy",
            "/tmp/policy.json",
            "--policy-sha256",
            "a" * 64,
            "--moment",
            "before",
        ])
        self.assertTrue(remote.remote)
        controller = guard.parse_args([
            "--plan",
            "/tmp/plan.json",
            "--plan-sha256",
            "b" * 64,
            "--moment",
            "after",
            "--receipt",
            "/tmp/receipt.json",
            "--execute",
            "--confirm",
            guard.CONFIRMATION,
        ])
        self.assertFalse(controller.remote)
        with self.assertRaises(guard.GuardError):
            guard.parse_args([
                "--plan",
                "/tmp/plan.json",
                "--plan-sha256",
                "b" * 64,
                "--moment",
                "after",
                "--receipt",
                "/tmp/receipt.json",
                "--execute",
                "--confirm",
                "WRONG",
            ])

    def test_bootstrap_and_phone_scan_are_single_argv_elements(self) -> None:
        self.assertNotIn("\n", guard.REMOTE_BOOTSTRAP)
        self.assertNotIn("\n", guard.PHONE_PROC_SCAN_SCRIPT)
        self.assertIn("base64.b64decode", guard.REMOTE_BOOTSTRAP)
        self.assertIn("toybox base64 -d", guard.PHONE_PROC_SCAN_SCRIPT)

    def test_remote_and_controller_success_reopen_all_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = executable_plan(root)
            runner = FakeCommandRunner(value["remote_policy"])
            remote_value, remote_raw = execute_remote_fixture(
                value,
                root / "remote-evidence",
                runner=runner,
            )
            guard.validate_remote_output(
                remote_value,
                value["remote_policy"],
                value["remote_policy_artifact"]["sha256"],
                "before",
            )
            receipt_value = execute_controller_fixture(
                value,
                root / "controller-evidence",
                remote_raw,
            )
            self.assertIs(
                guard.validate_receipt_evidence(receipt_value, value),
                receipt_value,
            )
            flattened = [item for argv in runner.calls for item in argv]
            self.assertNotIn("start-server", flattened)
            self.assertNotIn("kill-server", flattened)

    def test_remote_rejects_stale_policy_and_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = executable_plan(root)
            with self.assertRaises(guard.GuardError):
                guard.execute_remote(
                    value["remote_policy"],
                    "f" * 64,
                    "before",
                    output_root=root / "bad-policy",
                )
            helper = Path(value["remote_artifacts"]["helper"]["path"])
            helper.write_bytes(helper.read_bytes() + b"\n")
            with self.assertRaises(guard.GuardError):
                execute_remote_fixture(
                    value,
                    root / "bad-helper",
                )

    def test_remote_rejects_boot_process_listener_and_forward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = executable_plan(root)
            cases = [
                {
                    "boot_id": OP12_BOOT,
                    "runner": FakeCommandRunner(value["remote_policy"]),
                },
                {
                    "process_scanner": lambda unused: [{"pid": 99}],
                    "runner": FakeCommandRunner(value["remote_policy"]),
                },
                {
                    "listener_scanner": lambda unused: [{"port": 39312}],
                    "runner": FakeCommandRunner(value["remote_policy"]),
                },
                {
                    "runner": FakeCommandRunner(
                        value["remote_policy"],
                        forwards=(
                            value["phones"]["op12"]["wifi_selector"]
                            + " tcp:1 tcp:2\n"
                        ).encode("ascii"),
                    ),
                },
            ]
            for index, kwargs in enumerate(cases):
                with self.subTest(index=index):
                    with self.assertRaises(guard.GuardError):
                        execute_remote_fixture(
                            value,
                            root / f"rejected-{index}",
                            **kwargs,
                        )

    def test_phone_proc_match_and_wrong_hash_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = executable_plan(root)
            process = value["phones"]["op12"]["forbidden_processes"][0]
            stat_raw = (
                b"123 (worker) S "
                + b" ".join([b"1"] * 21)
                + b"\n"
            )
            cmdline_raw = process["executable_path"].encode("ascii") + b"\0"
            match = (
                "MATCH\t123\t"
                + process["executable_path"]
                + "\t"
                + process["sha256"]
                + "\t"
                + stat_raw.hex()
                + "\t"
                + cmdline_raw.hex()
                + "\n"
            ).encode("ascii")
            with self.assertRaises(guard.GuardError):
                execute_remote_fixture(
                    value,
                    root / "matching-process",
                    runner=FakeCommandRunner(
                        value["remote_policy"],
                        process_output=match,
                    ),
                )
            with self.assertRaises(guard.GuardError):
                execute_remote_fixture(
                    value,
                    root / "wrong-process-hash",
                    runner=FakeCommandRunner(
                        value["remote_policy"],
                        process_output=b"MISMATCH\n",
                        process_returncode=19,
                    ),
                )

    def test_command_bytes_are_rederived_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = executable_plan(root)
            remote_value, unused_raw = execute_remote_fixture(
                value,
                root / "remote-evidence",
            )
            del unused_raw
            forged = copy.deepcopy(remote_value)
            payload = forged["snapshots"]["op12"]
            snapshot_raw = base64.b64decode(
                payload["content_base64"],
                validate=True,
            )
            snapshot = guard.parse_json(snapshot_raw, "test.snapshot")
            snapshot["commands"][0]["stdout_base64"] = (
                base64.b64encode(b"offline\n").decode("ascii")
            )
            forged_raw = guard.canonical_bytes(snapshot)
            payload["content_base64"] = base64.b64encode(
                forged_raw
            ).decode("ascii")
            payload["artifact"]["bytes"] = len(forged_raw)
            payload["artifact"]["sha256"] = hashlib.sha256(
                forged_raw
            ).hexdigest()
            payload["artifact"]["stat"]["size"] = len(forged_raw)
            with self.assertRaises(guard.GuardError):
                guard.validate_remote_output(
                    forged,
                    value["remote_policy"],
                    value["remote_policy_artifact"]["sha256"],
                    "before",
                )

    def test_controller_timeout_and_orphan_are_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = executable_plan(root)
            remote_value, remote_raw = execute_remote_fixture(
                value,
                root / "remote-evidence",
            )
            del remote_value
            terminated = []

            def timeout_factory(argv, **unused):
                return FakePopen(argv, stdout=b"", timeout=True)

            with self.assertRaises(guard.GuardError):
                execute_controller_fixture(
                    value,
                    root / "timeout",
                    remote_raw,
                    popen_factory=timeout_factory,
                    terminator=lambda process: terminated.append(process.pid),
                )
            self.assertEqual(terminated, [4321])
            self.assertFalse((root / "timeout" / "receipt.json").exists())

            terminated.clear()
            with self.assertRaises(guard.GuardError):
                execute_controller_fixture(
                    value,
                    root / "orphan",
                    remote_raw,
                    group_absent_checker=lambda unused: False,
                    terminator=lambda process: terminated.append(process.pid),
                )
            self.assertEqual(terminated, [4321])
            self.assertFalse((root / "orphan" / "receipt.json").exists())

    def test_controller_rejects_partial_or_multi_packet_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = executable_plan(root)
            remote_value, remote_raw = execute_remote_fixture(
                value,
                root / "remote-evidence",
            )
            del remote_value
            packets = [
                base64.b64encode(remote_raw),
                base64.b64encode(remote_raw) + b"\nextra\n",
            ]
            for index, packet in enumerate(packets):
                with self.subTest(index=index):
                    factory = lambda argv, packet=packet, **unused: FakePopen(
                        argv,
                        stdout=packet,
                    )
                    with self.assertRaises(guard.GuardError):
                        execute_controller_fixture(
                            value,
                            root / f"packet-{index}",
                            remote_raw,
                            popen_factory=factory,
                        )

    def test_receipt_evidence_rejects_mutated_local_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = executable_plan(root)
            remote_value, remote_raw = execute_remote_fixture(
                value,
                root / "remote-evidence",
            )
            del remote_value
            receipt_value = execute_controller_fixture(
                value,
                root / "controller-evidence",
                remote_raw,
            )
            forged = copy.deepcopy(receipt_value)
            path = Path(
                forged["phones"]["op12"]["raw_snapshot_artifact"]["path"]
            )
            path.write_bytes(b"{}\n")
            forged["phones"]["op12"]["raw_snapshot_artifact"] = (
                guard.artifact_from_path(path, "test.forged_snapshot")
            )
            guard.validate_receipt(forged, value)
            with self.assertRaises(guard.GuardError):
                guard.validate_receipt_evidence(forged, value)


if __name__ == "__main__":
    unittest.main()
