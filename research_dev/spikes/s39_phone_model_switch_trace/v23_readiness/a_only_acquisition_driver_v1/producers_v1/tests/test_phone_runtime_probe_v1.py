#!/usr/bin/env python3

import copy
import hashlib
import importlib.util
import ipaddress
from pathlib import Path
import stat
import tempfile
import types
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "phone_runtime_probe_v1.py"
SPEC = importlib.util.spec_from_file_location("phone_runtime_probe_v1", SOURCE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


BOOT_ID = "11111111-1111-4111-8111-111111111111"
SERIAL = "3C15AU002CL00000"
SELECTOR = "172.20.173.218:5555"
WORKER = "/data/local/tmp/s39-v23/op15-worker/llama-layersplit"
RELAY = "/data/local/tmp/s39-v23/op15-relay/stage-direct-relay"
SHARD = "/data/local/tmp/s39-v23/qwen3-14b-op15.gguf"
WORKER_SHA = "a" * 64
RELAY_SHA = "e" * 64
SHARD_SHA = "b" * 64
MODEL_SHA = "c" * 64
PID = 123
TICKS = 456
LOCAL_IP = "192.168.1.2"
PEER_IP = "192.168.1.3"
LOCAL_PORT = 42000
PEER_PORT = 41000
SOCKET_INODE = 42
STAMP_NS = 1_704_067_200_000_000_000


def stat_row(size, inode):
    return {
        "ctime_ns": STAMP_NS,
        "device_id": 1,
        "inode": inode,
        "mode": stat.S_IFREG | 0o755,
        "mtime_ns": STAMP_NS,
        "size": size,
    }


def plan():
    argv = [
        WORKER,
        "-m",
        SHARD,
        "--mode",
        "stagenet",
        "--port",
        "40000",
    ]
    return {
        "android": {
            "adb_path": "/usr/bin/adb",
            "adb_port": 5038,
            "adb_selector": SELECTOR,
            "adb_sha256": "d" * 64,
            "boot_id_source": "phase_fresh_snapshot",
            "device": "OP611FL1",
            "model": "CPH2749",
            "physical_serial": SERIAL,
            "product": "CPH2749",
        },
        "capture_schema": probe.CAPTURE_SCHEMA,
        "model_id": "qwen3-14b-q4_k_m",
        "model_sha256": MODEL_SHA,
        "network_process": {
            "argv": argv,
            "artifact": {
                "bytes": 100,
                "path": WORKER,
                "sha256": WORKER_SHA,
                "stat": stat_row(100, 2),
            },
            "executable_path": WORKER,
            "role": "stagenet_worker",
        },
        "process": {
            "argv": argv,
            "executable_path": WORKER,
        },
        "schema": probe.PLAN_SCHEMA,
        "shard_artifact": {
            "bytes": 200,
            "path": SHARD,
            "sha256": SHARD_SHA,
            "stat": stat_row(200, 3),
        },
        "stage_v3": {
            "expected_active_sequences": 0,
            "source": "relay_owned_status",
        },
        "telemetry": {
            "direct_peer_ipv4": PEER_IP,
            "direct_peer_local_port": LOCAL_PORT,
            "direct_peer_port": PEER_PORT,
            "interface": "wlan0",
            "local_ipv4": LOCAL_IP,
            "max_gpu_millic": 80000,
            "min_available_bytes": 512 * 1024 * 1024,
        },
        "worker_artifact": {
            "bytes": 100,
            "path": WORKER,
            "sha256": WORKER_SHA,
            "stat": stat_row(100, 2),
        },
    }


def process_stat(pid=PID, ticks=TICKS):
    fields = ["S", *["0"] * 18, str(ticks), *["0"] * 3]
    return f"{pid} (llama worker) {' '.join(fields)}\n".encode("ascii")


def android_stat(value):
    return (
        f"DEV={value['device_id']}|INO={value['inode']}|"
        f"SIZE={value['size']}|MODE={value['mode']:x}|"
        "MTIME_S=1704067200|"
        "MTIME=2024-01-01 00:00:00.000000000 +0000|"
        "CTIME_S=1704067200|"
        "CTIME=2024-01-01 00:00:00.000000000 +0000"
    )


def tcp_hex(ipv4):
    return ipaddress.IPv4Address(ipv4).packed[::-1].hex().upper()


def remote_output(
    value=None,
    network_pid=PID,
    network_ticks=TICKS,
):
    value = value or plan()
    argv = value["process"]["argv"]
    cmdline = b"\x00".join(item.encode("ascii") for item in argv) + b"\x00"
    network = value["network_process"]
    network_cmdline = (
        b"\x00".join(item.encode("ascii") for item in network["argv"])
        + b"\x00"
    )
    tcp = (
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when "
        "retrnsmt   uid  timeout inode\n"
        f"  0: {tcp_hex(LOCAL_IP)}:{LOCAL_PORT:04X} "
        f"{tcp_hex(PEER_IP)}:{PEER_PORT:04X} 01 00000000:00000000 00:00000000 "
        f"00000000 0 0 {SOCKET_INODE}\n"
    ).encode("ascii")
    fields = {
        "SERIAL": SERIAL,
        "PRODUCT": "CPH2749",
        "MODEL": "CPH2749",
        "DEVICE": "OP611FL1",
        "BOOT": BOOT_ID,
        "EXE": WORKER.encode("ascii").hex(),
        "CMD": cmdline.hex(),
        "STAT": process_stat().hex(),
        "STAT2": process_stat().hex(),
        "NEXE": network["executable_path"].encode("ascii").hex(),
        "NCMD": network_cmdline.hex(),
        "NSTAT": process_stat(network_pid, network_ticks).hex(),
        "NSTAT2": process_stat(network_pid, network_ticks).hex(),
        "PSTATUS": b"Name:\tworker\nVmSwap:\t0 kB\n".hex(),
        "MEMINFO": (
            b"MemAvailable: 2097152 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n"
        ).hex(),
        "THERMAL": b"Thermal Status: 0\n".hex(),
        "ZONES": b"gpu-therm=45000\ncpu-therm=40000\n".hex(),
        "IPV4": (
            b"1: wlan0    inet 192.168.1.2/24 brd 192.168.1.255 scope global wlan0\n"
        ).hex(),
        "RX": "1000",
        "TX": "2000",
        "TCP": tcp.hex(),
        "FDS": f"socket:[{SOCKET_INODE}]\n".encode("ascii").hex(),
        "WORKERSTAT": android_stat(value["worker_artifact"]["stat"]),
        "SHARDSTAT": android_stat(value["shard_artifact"]["stat"]),
        "NETWORKSTAT": android_stat(network["artifact"]["stat"]),
    }
    order = [
        "SERIAL",
        "PRODUCT",
        "MODEL",
        "DEVICE",
        "BOOT",
        "EXE",
        "CMD",
        "STAT",
        "STAT2",
        "NEXE",
        "NCMD",
        "NSTAT",
        "NSTAT2",
        "PSTATUS",
        "MEMINFO",
        "THERMAL",
        "ZONES",
        "IPV4",
        "RX",
        "TX",
        "TCP",
        "FDS",
        "WORKERSTAT",
        "SHARDSTAT",
        "NETWORKSTAT",
    ]
    return "".join(f"{key} {fields[key]}\n" for key in order).encode("ascii")


def parse_snapshot(
    value,
    raw=None,
    pid=PID,
    ticks=TICKS,
    network_pid=PID,
    network_ticks=TICKS,
    boot_id=BOOT_ID,
):
    return probe.parse_remote_snapshot(
        remote_output(value) if raw is None else raw,
        value,
        pid,
        ticks,
        network_pid,
        network_ticks,
        boot_id,
    )


class PhoneRuntimeProbeTests(unittest.TestCase):
    def validated(self):
        return probe.validate_plan(copy.deepcopy(plan()))

    def test_plan_requires_pinned_wifi_adb_and_quiescent_stage(self):
        for field, value, message in (
            ("adb_port", 5037, "adb.port"),
            ("adb_selector", SERIAL, "E_ADB_SELECTOR"),
        ):
            with self.subTest(field=field):
                candidate = plan()
                candidate["android"][field] = value
                with self.assertRaisesRegex(probe.ProbeError, message):
                    probe.validate_plan(candidate)
        candidate = plan()
        candidate["stage_v3"]["expected_active_sequences"] = 1
        with self.assertRaisesRegex(probe.ProbeError, "expected_active_sequences"):
            probe.validate_plan(candidate)
        candidate = plan()
        candidate["capture_schema"] = "s39-unbound-capture-v1"
        with self.assertRaisesRegex(probe.ProbeError, "E_CAPTURE_SCHEMA"):
            probe.validate_plan(candidate)
        candidate = plan()
        candidate["android"]["boot_id_source"] = "pre_reboot_plan"
        with self.assertRaisesRegex(probe.ProbeError, "boot_id_source"):
            probe.validate_plan(candidate)
        candidate = plan()
        candidate["android"]["boot_id"] = BOOT_ID
        with self.assertRaisesRegex(probe.ProbeError, "E_KEYS"):
            probe.validate_plan(candidate)

    def test_remote_snapshot_accepts_exact_live_identity(self):
        value = self.validated()
        result = parse_snapshot(value)
        self.assertEqual(result["physical_serial"], SERIAL)
        self.assertEqual(result["boot_id"], BOOT_ID)
        self.assertEqual(result["process"]["pid"], PID)
        self.assertEqual(result["process"]["start_ticks"], TICKS)
        self.assertEqual(result["process_swap_bytes"], 0)
        self.assertEqual(result["system_swap_used_bytes"], 0)
        self.assertEqual(result["direct_peer"]["socket_inode"], SOCKET_INODE)
        self.assertEqual(result["gpu_max_millic"], 45000)
        with self.assertRaisesRegex(probe.ProbeError, "remote.boot_id"):
            parse_snapshot(
                value,
                boot_id="22222222-2222-4222-8222-222222222222",
            )

    def test_remote_snapshot_rejects_identity_and_hash_mutations(self):
        value = self.validated()
        valid = remote_output(value)
        mutations = {
            "serial": valid.replace(SERIAL.encode(), b"wrong-serial", 1),
            "boot": valid.replace(
                BOOT_ID.encode(),
                b"22222222-2222-4222-8222-222222222222",
                1,
            ),
            "pid": valid.replace(
                process_stat().hex().encode(),
                process_stat(pid=124).hex().encode(),
                1,
            ),
            "ticks": valid.replace(
                process_stat().hex().encode(),
                process_stat(ticks=457).hex().encode(),
                1,
            ),
            "exe": valid.replace(WORKER.encode().hex().encode(), b"2f77726f6e67", 1),
            "worker_stat": valid.replace(b"INO=2", b"INO=9", 1),
            "shard_stat": valid.replace(b"INO=3", b"INO=9", 1),
            "process_changed": valid.replace(
                b"STAT2 " + process_stat().hex().encode("ascii"),
                b"STAT2 " + process_stat(ticks=999).hex().encode("ascii"),
                1,
            ),
            "network_process_changed": valid.replace(
                b"NSTAT2 " + process_stat().hex().encode("ascii"),
                b"NSTAT2 " + process_stat(ticks=999).hex().encode("ascii"),
                1,
            ),
        }
        bad_plan = copy.deepcopy(value)
        bad_plan["process"]["argv"] = [*value["process"]["argv"], "--extra"]
        mutations["argv"] = remote_output(bad_plan)
        for name, raw in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(probe.ProbeError):
                    parse_snapshot(value, raw)
        for name, pid, ticks in (
            ("pid_arg", PID + 1, TICKS),
            ("ticks_arg", PID, TICKS + 1),
        ):
            with self.subTest(name=name):
                with self.assertRaises(probe.ProbeError):
                    parse_snapshot(value, valid, pid=pid, ticks=ticks)
        for name, network_pid, network_ticks in (
            ("network_pid_arg", PID + 1, TICKS),
            ("network_ticks_arg", PID, TICKS + 1),
        ):
            with self.subTest(name=name):
                with self.assertRaises(probe.ProbeError):
                    parse_snapshot(
                        value,
                        valid,
                        network_pid=network_pid,
                        network_ticks=network_ticks,
                    )

    def test_direct_peer_is_owned_by_bound_network_process(self):
        value = plan()
        value["network_process"] = {
            "argv": [RELAY, "--listen", "41000"],
            "artifact": {
                "bytes": 300,
                "path": RELAY,
                "sha256": RELAY_SHA,
                "stat": stat_row(300, 4),
            },
            "executable_path": RELAY,
            "role": "direct_relay",
        }
        value = probe.validate_plan(value)
        raw = remote_output(value, network_pid=789, network_ticks=1011)
        result = parse_snapshot(
            value,
            raw,
            network_pid=789,
            network_ticks=1011,
        )
        self.assertEqual(result["network_process"]["role"], "direct_relay")
        self.assertEqual(result["network_process"]["pid"], 789)
        self.assertEqual(
            result["network_process"]["executable_sha256"],
            RELAY_SHA,
        )
        wrong_network = raw.replace(
            RELAY.encode("ascii").hex().encode("ascii"),
            b"2f77726f6e67",
            1,
        )
        with self.assertRaisesRegex(probe.ProbeError, "network_executable"):
            parse_snapshot(
                value,
                wrong_network,
                network_pid=789,
                network_ticks=1011,
            )

    def test_worker_swap_is_zero_but_nonzero_system_swap_is_recorded(self):
        value = self.validated()
        raw = remote_output(value)
        process_swap = raw.replace(
            b"VmSwap:\t0 kB\n".hex().encode(),
            b"VmSwap:\t1 kB\n".hex().encode(),
            1,
        )
        with self.assertRaisesRegex(probe.ProbeError, "process_swap_bytes"):
            parse_snapshot(value, process_swap)

        system_swap = raw.replace(
            (
                b"MemAvailable: 2097152 kB\n"
                b"SwapTotal: 0 kB\nSwapFree: 0 kB\n"
            ).hex().encode(),
            (
                b"MemAvailable: 2097152 kB\n"
                b"SwapTotal: 2097152 kB\nSwapFree: 1048576 kB\n"
            ).hex().encode(),
            1,
        )
        result = parse_snapshot(value, system_swap)
        self.assertEqual(result["system_swap_used_bytes"], 1024 * 1024 * 1024)

    def test_capture_schema_is_bound_and_allowlisted(self):
        value = self.validated()
        remote = parse_snapshot(value)
        self.assertEqual(probe.capture_row(value, remote)["schema"], probe.CAPTURE_SCHEMA)
        value["capture_schema"] = probe.CAPTURE_SCHEMA_V24
        self.assertEqual(
            probe.capture_row(probe.validate_plan(value), remote)["schema"],
            probe.CAPTURE_SCHEMA_V24,
        )

    def test_missing_or_foreign_direct_peer_is_rejected(self):
        value = self.validated()
        raw = remote_output(value)
        missing_fd = raw.replace(
            f"socket:[{SOCKET_INODE}]\n".encode().hex().encode(),
            b"".hex().encode(),
            1,
        )
        with self.assertRaisesRegex(probe.ProbeError, "E_DIRECT_PEER"):
            parse_snapshot(value, missing_fd)
        wrong_peer = copy.deepcopy(value)
        wrong_peer["telemetry"]["direct_peer_ipv4"] = "192.168.1.9"
        with self.assertRaisesRegex(probe.ProbeError, "E_DIRECT_PEER"):
            parse_snapshot(wrong_peer, raw)

    def test_capture_row_matches_phone_route_schema(self):
        value = self.validated()
        remote = parse_snapshot(value)
        row = probe.capture_row(value, remote)
        self.assertEqual(row["schema"], probe.CAPTURE_SCHEMA)
        self.assertEqual(row["active_sequences"], 0)
        self.assertEqual(row["worker_pid"], PID)
        self.assertEqual(row["worker_start_ticks"], TICKS)
        self.assertEqual(
            set(row),
            {
                "active_sequences",
                "available_bytes",
                "boot_id",
                "device",
                "direct_peer",
                "gpu_max_millic",
                "interface",
                "loaded_shard_path",
                "loaded_shard_sha256",
                "model",
                "model_id",
                "model_sha256",
                "network_executable_path",
                "network_executable_sha256",
                "network_pid",
                "network_process_role",
                "network_start_ticks",
                "process_swap_bytes",
                "product",
                "schema",
                "serial",
                "system_swap_used_bytes",
                "worker_executable_path",
                "worker_executable_sha256",
                "worker_pid",
                "worker_start_ticks",
            },
        )

    def test_probe_never_opens_stagenet_or_hashes_large_artifacts(self):
        value = self.validated()
        argv = probe.remote_argv(value, PID, PID)
        command = argv[-1]
        self.assertNotIn("STAGE_V3", command)
        self.assertNotIn("sha256sum", command)
        self.assertIn("WORKERSTAT", command)
        self.assertIn("SHARDSTAT", command)
        self.assertNotIn("socket.create_connection", SOURCE.read_text(encoding="ascii"))

    def test_occupied_worker_needs_no_second_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            adb = Path(directory) / "adb"
            adb.write_bytes(b"adb-client")
            adb.chmod(0o755)
            value = plan()
            value["android"]["adb_path"] = str(adb)
            value["android"]["adb_sha256"] = hashlib.sha256(
                b"adb-client"
            ).hexdigest()
            value = probe.validate_plan(value)

            class Runner:
                def __init__(self):
                    self.calls = []

                def run(self, argv, *, timeout):
                    self.calls.append((list(argv), timeout))
                    return types.SimpleNamespace(
                        returncode=0,
                        stdout=remote_output(value),
                        stderr=b"",
                    )

            runner = Runner()
            row = probe.run_probe(
                value,
                runner,
                PID,
                TICKS,
                PID,
                TICKS,
                BOOT_ID,
                True,
            )
            self.assertEqual(row["schema"], probe.CAPTURE_SCHEMA)
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(runner.calls[0][0][:5], [
                str(adb),
                "-P",
                "5038",
                "-s",
                SELECTOR,
            ])

    def test_inline_plan_is_canonical_and_digest_bound(self):
        value = plan()
        raw = probe.canonical_compact(value)
        checksum = hashlib.sha256(raw).hexdigest()
        parsed = probe.parse_plan_json(raw.decode("ascii"), checksum)
        self.assertEqual(parsed["model_id"], value["model_id"])
        with self.assertRaisesRegex(probe.ProbeError, "plan.sha256"):
            probe.parse_plan_json(raw.decode("ascii"), "0" * 64)
        pretty = __import__("json").dumps(value, sort_keys=True)
        with self.assertRaisesRegex(probe.ProbeError, "plan.canonical"):
            probe.parse_plan_json(
                pretty,
                hashlib.sha256(pretty.encode("ascii")).hexdigest(),
            )

    def test_adb_selector_is_explicit_in_remote_argv(self):
        value = self.validated()
        argv = probe.remote_argv(value, PID, PID)
        self.assertEqual(
            argv[:5],
            ["/usr/bin/adb", "-P", "5038", "-s", SELECTOR],
        )
        self.assertIn(str(PID), argv[-1])
        self.assertIn(WORKER, argv[-1])
        self.assertIn(SHARD, argv[-1])

    def test_adb_binary_hash_is_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adb"
            path.write_bytes(b"adb-client")
            path.chmod(0o755)
            android = copy.deepcopy(plan()["android"])
            android["adb_path"] = str(path)
            android["adb_sha256"] = hashlib.sha256(b"adb-client").hexdigest()
            probe.read_sealed_adb(android)
            android["adb_sha256"] = "0" * 64
            with self.assertRaisesRegex(probe.ProbeError, "adb.sha256"):
                probe.read_sealed_adb(android)


if __name__ == "__main__":
    unittest.main()
