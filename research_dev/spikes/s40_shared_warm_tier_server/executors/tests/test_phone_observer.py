#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
EXECUTORS = HERE.parent
if str(EXECUTORS) not in sys.path:
    sys.path.insert(0, str(EXECUTORS))

from a6000_phone_observer import (
    PhoneObserver,
    parse_interfaces,
    parse_meminfo,
    parse_tcp,
    parse_temperatures,
    parse_thermal_status,
    reciprocal_peer,
)
from phone_gateway import GatewayError, canonical_bytes
from validate_phone_observer import argv_sha256, validate_phone_observer


OP12_BOOT = "11111111-1111-4111-8111-111111111111"
OP15_BOOT = "22222222-2222-4222-8222-222222222222"
OP12_ADDRESS = "192.168.1.12"
OP15_ADDRESS = "192.168.1.15"
OP12_PORT = 41000
OP15_PORT = 5000
RUN_ID = "observer-run"
MODEL_ID = "qwen3-14b-q8"
SSH_SHA = "a" * 64


def remote_package() -> dict:
    roles = (
        "a6000_phone_observer",
        "a6000_phone_route_control",
        "adb",
        "phone_gateway",
        "python",
        "readiness_v23",
        "remote_config",
    )
    return {
        "a6000_identity": "b" * 64,
        "files": [
            {
                "bytes": index + 1,
                "path": f"/remote/{role}",
                "role": role,
                "sha256": f"{index + 1:x}" * 64,
            }
            for index, role in enumerate(roles)
        ],
        "host_boot_id": "33333333-3333-4333-8333-333333333333",
        "schema": "s40-a6000-identity-package-v1",
    }


def local_package() -> dict:
    roles = (
        "identity_public_key",
        "known_hosts",
        "ssh_executable",
        "ssh_keygen",
    )
    return {
        "files": [
            {
                "bytes": index + 1,
                "path": f"/local/{role}",
                "role": role,
                "sha256": f"{index + 8:x}" * 64,
            }
            for index, role in enumerate(roles)
        ],
        "identity_file_path": "/local/id_ed25519",
        "identity_public_key_fingerprint": "SHA256:fixed-test-key",
        "schema": "s40-local-ssh-identity-package-v1",
        "ssh_argv": ["/usr/bin/ssh", "a6000"],
        "ssh_env": {
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    }


def socket_rows(name: str) -> list[dict]:
    if name == "op12":
        return [{
            "local_address": OP12_ADDRESS,
            "local_port": OP12_PORT,
            "remote_address": OP15_ADDRESS,
            "remote_port": OP15_PORT,
            "socket_inode": 111,
        }]
    return [{
        "local_address": OP15_ADDRESS,
        "local_port": OP15_PORT,
        "remote_address": OP12_ADDRESS,
        "remote_port": OP12_PORT,
        "socket_inode": 222,
    }]


def snapshot(
    name: str,
    started_ns: int,
    completed_ns: int,
    counter: int,
    *,
    sockets: bool = True,
) -> dict:
    is_op12 = name == "op12"
    runtime_root = f"/{name}/runtime"
    runtime_names = {
        "libcxx_shared": "libc++_shared.so",
        "libggml": "libggml.so",
        "libggml_base": "libggml-base.so",
        "libggml_cpu": "libggml-cpu.so",
        "libggml_hexagon": "libggml-hexagon.so",
        "libggml_opencl": "libggml-opencl.so",
        "libllama": "libllama.so",
        "libllama_common": "libllama-common.so",
        "worker": "llama-layersplit",
    }
    if is_op12:
        runtime_names["htp_skel_v75"] = "libggml-htp-v75.so"
    else:
        runtime_names["htp_skel_v81"] = "libggml-htp-v81.so"
        runtime_names["relay"] = "llama-stage-direct-relay"
    runtime_files = []
    for index, (role, filename) in enumerate(sorted(runtime_names.items()), 1):
        runtime_files.append({
            "path": f"{runtime_root}/{filename}",
            "role": role,
            "sha256": f"{index:x}" * 64,
            "stat": {
                "ctime_ns": 1,
                "device_id": 2,
                "inode": 100 + index,
                "mode": 0o100555,
                "mtime_ns": 5,
                "size": 1000 + index,
            },
        })
    runtime_by_role = {row["role"]: row for row in runtime_files}
    processes = []
    if sockets:
        layer_start = 30 if is_op12 else 0
        layer_end = 40 if is_op12 else 30
        port = OP12_PORT if is_op12 else 40000
        mode = "tailv3" if is_op12 else "stagenet"
        worker = runtime_by_role["worker"]
        stage_argv = [
            worker["path"],
            "-m",
            f"/{name}/weights.gguf",
            "--mode",
            mode,
            "--port",
            str(port),
            "--driver-batch",
            "8",
            "--driver-context",
            "64",
            "--driver-max-prefill",
            "8",
            "--devices",
            "GPUOpenCL",
            "-ngl",
            "99",
        ]
        processes.append({
            "argv": stage_argv,
            "artifact_path": worker["path"],
            "artifact_role": "worker",
            "artifact_sha256": worker["sha256"],
            "backend": "GPUOpenCL",
            "cmdline_sha256": argv_sha256(stage_argv),
            "env": {
                "ADSP_LIBRARY_PATH": runtime_root,
                "LAYERSPLIT_MODEL_SHA256": "a" * 64,
                "LAYERSPLIT_PLACEMENT_CERT": "1",
                "LD_LIBRARY_PATH": runtime_root,
                "LLAMA_LAYER_END": str(layer_end),
                "LLAMA_LAYER_START": str(layer_start),
                "PATH": "/system/bin:/system/xbin",
            },
            "head_host": None,
            "head_port": None,
            "kind": "STAGE_TAIL" if is_op12 else "STAGE_HEAD",
            "layer_end": layer_end,
            "layer_start": layer_start,
            "listen_port": port,
            "name": "stage",
            "pid": 1200 if is_op12 else 1500,
            "process_start_ticks": 12000 if is_op12 else 15000,
            "socket_inodes": [111] if is_op12 else [333],
            "tail_host": None,
            "tail_port": None,
            "tail_source_port": None,
        })
        if not is_op12:
            relay = runtime_by_role["relay"]
            relay_argv = [
                relay["path"],
                "--listen",
                "42000",
                "--head",
                "127.0.0.1:40000",
                "--tail",
                f"{OP12_ADDRESS}:{OP12_PORT}",
                "--tail-source-port",
                str(OP15_PORT),
            ]
            processes.append({
                "argv": relay_argv,
                "artifact_path": relay["path"],
                "artifact_role": "relay",
                "artifact_sha256": relay["sha256"],
                "backend": "NETWORK",
                "cmdline_sha256": argv_sha256(relay_argv),
                "env": {
                    "LD_LIBRARY_PATH": runtime_root,
                    "PATH": "/system/bin:/system/xbin",
                },
                "head_host": "127.0.0.1",
                "head_port": 40000,
                "kind": "DIRECT_RELAY",
                "layer_end": None,
                "layer_start": None,
                "listen_port": 42000,
                "name": "relay",
                "pid": 1501,
                "process_start_ticks": 15001,
                "socket_inodes": [222],
                "tail_host": OP12_ADDRESS,
                "tail_port": OP12_PORT,
                "tail_source_port": OP15_PORT,
            })
    return {
        "available_bytes": 1024 * 1024 * 1024,
        "boot_id": OP12_BOOT if is_op12 else OP15_BOOT,
        "completed_ns": completed_ns,
        "device": "OP595DL1" if is_op12 else "OP611FL1",
        "interfaces": {
            "wlan0": {
                "ipv4": [OP12_ADDRESS if is_op12 else OP15_ADDRESS],
                "rx_bytes": counter,
                "tx_bytes": counter + 10,
            },
        },
        "model": "CPH2583" if is_op12 else "CPH2749",
        "product": "CPH2583" if is_op12 else "CPH2749",
        "processes": processes,
        "runtime_files": runtime_files,
        "schema": "s40-phone-runtime-snapshot-v1",
        "serial": "5ae7a43d" if is_op12 else "3C15AU002CL00000",
        "started_ns": started_ns,
        "swap_total_bytes": 1024,
        "swap_used_bytes": 0,
        "tcp_established": socket_rows(name) if sockets else [],
        "temperatures": [{"name": "soc", "temp_millic": 41000}],
        "thermal_status": 0,
    }


def identity(
    started_ns: int,
    completed_ns: int,
    counter: int,
) -> dict:
    package = remote_package()
    return {
        "a6000_identity": "b" * 64,
        "completed_ns": completed_ns,
        "direct_peer": {
            "op12_address": OP12_ADDRESS,
            "op12_port": OP12_PORT,
            "op12_socket_inode": 111,
            "op15_address": OP15_ADDRESS,
            "op15_port": OP15_PORT,
            "op15_socket_inode": 222,
            "schema": "s40-phone-direct-peer-v2",
        },
        "model_id": MODEL_ID,
        "phones": {
            "op12": snapshot(
                "op12",
                started_ns + 1,
                started_ns + 10,
                counter,
            ),
            "op15": snapshot(
                "op15",
                started_ns + 11,
                started_ns + 20,
                counter,
            ),
        },
        "remote_identity_package": package,
        "remote_identity_package_sha256": hashlib.sha256(
            canonical_bytes(package)
        ).hexdigest(),
        "route_instance_id": "route-instance-a",
        "schema": "s40-phone-identity-v2",
        "started_ns": started_ns,
    }


def bridge_argv(action: str) -> list[str]:
    result = [
        "ssh",
        "a6000",
        "/usr/bin/python3",
        "-B",
        "-s",
        "/abs/a6000_phone_observer.py",
        "--action",
        action,
        "--config",
        "/abs/route-config.json",
        "--model",
        MODEL_ID,
    ]
    if action == "telemetry":
        result.extend(["--run-id", RUN_ID, "--interval-ms", "1000"])
    return result


class Fixture:
    def __init__(self, test: unittest.TestCase):
        temporary = tempfile.TemporaryDirectory()
        test.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.identity_path = self.root / "identity.json"
        self.telemetry_path = self.root / "telemetry.jsonl"
        self.stderr_path = self.root / "telemetry.stderr"
        self.stderr_path.write_bytes(b"")
        local = local_package()
        local_sha = hashlib.sha256(canonical_bytes(local)).hexdigest()
        remote_sha = identity(1, 2, 1)[
            "remote_identity_package_sha256"
        ]
        self.identity = {
            "argv": bridge_argv("identity"),
            "completed_ns": 1100,
            "exit_code": 0,
            "local_identity_package": local,
            "local_identity_package_sha256": local_sha,
            "remote": identity(100, 160, 1000),
            "remote_identity_package_sha256": remote_sha,
            "run_id": RUN_ID,
            "schema": "s40-phone-identity-bridge-v2",
            "ssh_config_sha256": SSH_SHA,
            "started_ns": 1000,
            "success": True,
        }
        header = {
            "identity": identity(200, 260, 1100),
            "interval_ms": 1000,
            "run_id": RUN_ID,
            "schema": "s40-phone-telemetry-header-v1",
        }
        sample = {
            "active_route": {
                "direct_peer": identity(1, 2, 1)["direct_peer"],
                "model_id": MODEL_ID,
                "route_instance_id": "route-instance-a",
                "schema": "s40-phone-route-observation-v1",
            },
            "completed_ns": 360,
            "phones": {
                "op12": snapshot("op12", 301, 310, 1200),
                "op15": snapshot("op15", 311, 320, 1200),
            },
            "run_id": RUN_ID,
            "sample_index": 1,
            "schema": "s40-phone-telemetry-sample-v2",
            "started_ns": 300,
        }
        footer = {
            "active_route": None,
            "completed_ns": 460,
            "phones": {
                "op12": snapshot(
                    "op12",
                    401,
                    410,
                    1300,
                    sockets=False,
                ),
                "op15": snapshot(
                    "op15",
                    411,
                    420,
                    1300,
                    sockets=False,
                ),
            },
            "run_id": RUN_ID,
            "sample_count": 1,
            "schema": "s40-phone-telemetry-footer-v2",
            "started_ns": 400,
        }
        argv = bridge_argv("telemetry")
        self.rows = [
            {
                "local_identity_package": local,
                "local_identity_package_sha256": local_sha,
                "local_received_ns": received,
                "remote": remote,
                "remote_argv": argv,
                "remote_identity_package_sha256": remote_sha,
                "run_id": RUN_ID,
                "schema": "s40-phone-telemetry-bridge-v2",
                "ssh_config_sha256": SSH_SHA,
            }
            for received, remote in (
                (1200, header),
                (1300, sample),
                (1400, footer),
            )
        ]
        self.write()

    def write(self) -> None:
        self.identity_path.write_bytes(canonical_bytes(self.identity))
        self.telemetry_path.write_bytes(
            b"".join(canonical_bytes(row) for row in self.rows)
        )

    def validate(self) -> dict:
        return validate_phone_observer(
            self.identity_path,
            self.telemetry_path,
            self.stderr_path,
            MODEL_ID,
            RUN_ID,
        )


class PhoneObserverTests(unittest.TestCase):
    def test_current_state_only_treats_absence_as_inactive(self):
        observer = PhoneObserver.__new__(PhoneObserver)

        def missing():
            raise FileNotFoundError("absent")

        observer.active_state = missing
        self.assertIsNone(observer.current_state())
        for error in (
            GatewayError("corrupt journal"),
            PermissionError("unreadable journal"),
        ):
            with self.subTest(error=type(error).__name__):
                def failed(error=error):
                    raise error

                observer.active_state = failed
                with self.assertRaises(type(error)):
                    observer.current_state()

    def test_parsers(self):
        self.assertEqual(
            parse_meminfo(
                "MemAvailable: 1048576 kB\n"
                "SwapTotal: 4096 kB\n"
                "SwapFree: 4096 kB\n"
            ),
            {
                "available_bytes": 1073741824,
                "swap_total_bytes": 4194304,
                "swap_used_bytes": 0,
            },
        )
        self.assertEqual(
            parse_temperatures("soc|41000\nbattery|35000\n"),
            [
                {"name": "battery", "temp_millic": 35000},
                {"name": "soc", "temp_millic": 41000},
            ],
        )
        self.assertEqual(
            parse_thermal_status("Thermal Status: 0\n"),
            0,
        )
        interfaces = parse_interfaces(
            "wlan0: 100 0 0 0 0 0 0 0 200 0 0 0 0 0 0 0\n",
            "1: wlan0 inet 192.168.1.12/24 brd 192.168.1.255\n",
        )
        self.assertEqual(
            interfaces["wlan0"],
            {
                "ipv4": ["192.168.1.12"],
                "rx_bytes": 100,
                "tx_bytes": 200,
            },
        )
        tcp = parse_tcp(
            "sl local_address rem_address st\n"
            "0: 0C01A8C0:A028 0F01A8C0:1388 01 "
            "00000000:00000000 00:00000000 00000000 1000 0 111\n"
        )
        self.assertEqual(tcp, socket_rows("op12"))

    def test_reciprocal_peer(self):
        expected = {
            "op12_address": OP12_ADDRESS,
            "op12_port": OP12_PORT,
            "op15_address": OP15_ADDRESS,
            "op15_port": OP15_PORT,
            "schema": "s40-phone-direct-peer-v2",
        }
        self.assertEqual(
            reciprocal_peer(
                snapshot("op12", 1, 2, 1),
                snapshot("op15", 3, 4, 1),
                expected,
            )["schema"],
            "s40-phone-direct-peer-v2",
        )
        expected["op12_port"] += 1
        with self.assertRaisesRegex(RuntimeError, "expected"):
            reciprocal_peer(
                snapshot("op12", 1, 2, 1),
                snapshot("op15", 3, 4, 1),
                expected,
            )

    def test_valid_evidence(self):
        fixture = Fixture(self)
        result = fixture.validate()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["boot_ids"],
            {"op12": OP12_BOOT, "op15": OP15_BOOT},
        )
        self.assertEqual(
            result["interface_byte_deltas"]["op12"]["wlan0"]["rx_bytes"],
            300,
        )

    def test_later_route_is_observed_and_model_bound(self):
        fixture = Fixture(self)
        second = copy.deepcopy(fixture.rows[1])
        second["local_received_ns"] = 1350
        remote = second["remote"]
        remote["sample_index"] = 2
        remote["started_ns"] = 370
        remote["completed_ns"] = 390
        remote["active_route"]["model_id"] = "qwen3-8b-q8"
        remote["active_route"]["route_instance_id"] = "route-instance-b"
        remote["phones"]["op12"]["started_ns"] = 371
        remote["phones"]["op12"]["completed_ns"] = 378
        remote["phones"]["op15"]["started_ns"] = 379
        remote["phones"]["op15"]["completed_ns"] = 386
        fixture.rows.insert(2, second)
        fixture.rows[-1]["remote"]["sample_count"] = 2
        fixture.write()
        result = fixture.validate()
        self.assertEqual(
            [
                (row["model_id"], row["route_instance_id"])
                for row in result["observed_routes"]
            ],
            [
                (MODEL_ID, "route-instance-a"),
                ("qwen3-8b-q8", "route-instance-b"),
            ],
        )

        fixture = Fixture(self)
        second = copy.deepcopy(fixture.rows[1])
        second["local_received_ns"] = 1350
        remote = second["remote"]
        remote["sample_index"] = 2
        remote["started_ns"] = 370
        remote["completed_ns"] = 390
        remote["active_route"]["model_id"] = "qwen3-8b-q8"
        remote["phones"]["op12"]["started_ns"] = 371
        remote["phones"]["op12"]["completed_ns"] = 378
        remote["phones"]["op15"]["started_ns"] = 379
        remote["phones"]["op15"]["completed_ns"] = 386
        fixture.rows.insert(2, second)
        fixture.rows[-1]["remote"]["sample_count"] = 2
        fixture.write()
        with self.assertRaisesRegex(RuntimeError, "changed model"):
            fixture.validate()

    def test_mutations_fail_closed(self):
        def add_process_flag(fixture):
            process = fixture.rows[1]["remote"]["phones"]["op12"][
                "processes"
            ][0]
            process["argv"].append("--verbose")
            process["cmdline_sha256"] = argv_sha256(process["argv"])

        def change_relay_source(fixture):
            process = fixture.rows[1]["remote"]["phones"]["op15"][
                "processes"
            ][1]
            process["tail_source_port"] = OP15_PORT + 1
            process["argv"][-1] = str(OP15_PORT + 1)
            process["cmdline_sha256"] = argv_sha256(process["argv"])

        def swap_runtime_role_path(fixture):
            files = fixture.rows[1]["remote"]["phones"]["op12"][
                "runtime_files"
            ]
            worker = next(row for row in files if row["role"] == "worker")
            worker["path"] = "/op12/runtime/libggml.so"

        mutations = {
            "bool exit": lambda fixture: fixture.identity.__setitem__(
                "exit_code",
                False,
            ),
            "stale boot": lambda fixture: fixture.rows[1]["remote"]["phones"][
                "op12"
            ].__setitem__(
                "boot_id",
                "33333333-3333-4333-8333-333333333333",
            ),
            "active peer missing": lambda fixture: fixture.rows[1]["remote"][
                "phones"
            ]["op12"].__setitem__("tcp_established", []),
            "counter reset": lambda fixture: fixture.rows[2]["remote"]["phones"][
                "op12"
            ]["interfaces"]["wlan0"].__setitem__("rx_bytes", 1),
            "footer active": lambda fixture: fixture.rows[2]["remote"].__setitem__(
                "active_route",
                fixture.rows[1]["remote"]["active_route"],
            ),
            "invalid IPv4": lambda fixture: fixture.identity["remote"]["phones"][
                "op12"
            ]["interfaces"]["wlan0"].__setitem__(
                "ipv4",
                ["999.168.1.12"],
            ),
            "wrong argv": lambda fixture: fixture.rows[0][
                "remote_argv"
            ].__setitem__(-5, "other-model"),
            "time reversal": lambda fixture: fixture.rows[1]["remote"].__setitem__(
                "started_ns",
                250,
            ),
            "remote dependency mutation": lambda fixture: fixture.identity[
                "remote"
            ]["remote_identity_package"]["files"][0].__setitem__(
                "sha256",
                "f" * 64,
            ),
            "local dependency mutation": lambda fixture: fixture.rows[0][
                "local_identity_package"
            ]["files"][0].__setitem__("bytes", 99),
            "coordinated extra process flag": add_process_flag,
            "relay source mismatch": change_relay_source,
            "socket not owned": lambda fixture: fixture.rows[1]["remote"][
                "phones"
            ]["op12"]["processes"][0].__setitem__("socket_inodes", []),
            "cross-phone model mismatch": lambda fixture: fixture.rows[1][
                "remote"
            ]["phones"]["op12"]["processes"][0]["env"].__setitem__(
                "LAYERSPLIT_MODEL_SHA256",
                "f" * 64,
            ),
            "runtime role path": swap_runtime_role_path,
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                fixture = Fixture(self)
                mutate(fixture)
                fixture.write()
                with self.assertRaises(RuntimeError):
                    fixture.validate()

    def test_stderr_fails_closed(self):
        fixture = Fixture(self)
        fixture.stderr_path.write_text("ssh warning\n", encoding="ascii")
        with self.assertRaisesRegex(RuntimeError, "stderr"):
            fixture.validate()


if __name__ == "__main__":
    unittest.main()
