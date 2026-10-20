"""WS16: matched per-device weight-stream microbenchmark (harness, protocol driver, energy convention)."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import threading
import unittest

from research_dev.scheduler.campaigns.burstgpt.tools import weight_stream_microbench as wsm


def base_config(**changes):
    config = {
        "schema": wsm.SCHEMA, "rows": [1, 2, 4], "segment_seconds": 10, "idle_s": 5, "cooldown_s": 0,
        "cadence": {"calls_per_token": 3, "gap_call_ms": 5, "gap_token_ms": 300, "warmup_tokens": 2},
        "legs": [
            {"name": "cpu-ffn", "kind": "desktop", "op": "ffn", "bench": "/x/bench", "model": "/m.gguf",
             "backend": "CPU", "threads": 8, "layer": 18},
            {"name": "pixel-ffn", "kind": "phone", "op": "ffn", "device": "pixel", "transport": "aoa-bridge",
             "protocol_version": 6, "activation": "swiglu", "layer": 18, "n_embd": 8, "n_ff": 64,
             "est_call_ms": {"1": 8.0, "2": 15.0, "4": 32.0}, "evidence": "/e.json"},
        ],
    }
    config.update(changes)
    return config


class FakeWorker(threading.Thread):
    """One-connection TCP worker speaking HELLO v5 or v6 and echoing EXECUTE payloads."""

    def __init__(self, version: int):
        super().__init__(daemon=True)
        self.version = version
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        self.hello_bytes = None
        self.executes = []
        self.shutdown = None

    def run(self):
        connection, _ = self.listener.accept()
        self.listener.close()
        with connection:
            size = 64 if self.version == 6 else 32
            self.hello_bytes = wsm.receive(connection, size)
            response = bytearray(96 if self.version == 6 else 64)
            struct.pack_into("<IHHH", response, 0, wsm.PROTOCOL_MAGIC, self.version, 2, 0)
            connection.sendall(bytes(response))
            while True:
                try:
                    header = wsm.receive(connection, wsm.EXECUTE.size)
                except RuntimeError:
                    return
                fields = wsm.EXECUTE.unpack(header)
                if fields[3] == 0:  # the USB bridge's local shutdown request: no reply
                    self.shutdown = fields
                    continue
                payload = wsm.receive(connection, fields[6])
                self.executes.append(fields)
                connection.sendall(wsm.EXECUTE_RESPONSE.pack(
                    wsm.PROTOCOL_MAGIC, self.version, 4, 0, 0, fields[3], fields[4], fields[5], fields[6],
                    wsm.fnv32(payload), fields[8], fields[9], 1234) + payload)


class Clock:
    def __init__(self):
        self.t = 1_000_000_000

    def __call__(self):
        self.t += 1_000_000  # 1 ms per read
        return self.t

    def sleep(self, seconds):
        self.t += int(seconds * 1e9)


class DriverTests(unittest.TestCase):
    def drive(self, version):
        config = base_config()
        leg = dict(config["legs"][1], protocol_version=version)
        worker = FakeWorker(version)
        worker.start()
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "LEG.jsonl"
            sent = wsm.drive_phone_leg(worker.port, config, leg, "sha256:" + "ab" * 32, 4, out,
                                       clock=clock, sleep=clock.sleep)
            records = wsm.read_jsonl(out)
        worker.join(5)
        return config, leg, worker, sent, records

    def test_v6_and_v5_hello_and_exact_budget(self):
        for version in (6, 5):
            config, leg, worker, sent, records = self.drive(version)
            self.assertEqual(len(worker.hello_bytes), 64 if version == 6 else 32)
            magic, got_version, message, mask, n_embd, n_ff, flags, max_tokens = struct.unpack_from(
                "<IHHQIIHH", worker.hello_bytes)
            self.assertEqual((got_version, mask, n_embd, n_ff, flags, max_tokens),
                             (version, 1 << 18, 8, 64, wsm.F16_IO_FLAG | wsm.SWIGLU_FLAG, 4))
            if version == 6:
                self.assertEqual(worker.hello_bytes[28:60], bytes.fromhex("ab" * 32))  # after the 28-byte header
            # every warm-up and measured call + the closing call = the worker's budget
            self.assertEqual(sent, wsm.phone_budget(config, leg))
            self.assertEqual(len(worker.executes), sent)
            kinds = [r["kind"] for r in records]
            self.assertEqual(kinds[0], "meta")
            self.assertEqual([r["phase"] for r in records if r["kind"] == "idle"], ["before", "after"])
            segments = [r for r in records if r["kind"] == "segment"]
            self.assertEqual([s["rows"] for s in segments], [1, 2, 4])
            for segment in segments:
                tokens = wsm.phone_tokens(config, leg, segment["rows"])
                self.assertEqual(segment["calls"], tokens * 3)
                calls = [r for r in records if r["kind"] == "call" and r["rows"] == segment["rows"]
                         and not r["warmup"]]
                self.assertEqual(len(calls), segment["calls"])
                self.assertTrue(all(segment["t0_ns"] <= c["t_start_ns"] <= segment["t1_ns"] for c in calls))
                self.assertEqual({c["compute_us"] for c in calls}, {1234})
            rows_in_requests = [fields[9] for fields in worker.executes]
            self.assertEqual(rows_in_requests[-1], 1)  # closing call
            self.assertEqual(records[0]["logical_bytes"], 3 * 8 * 64 * 2)

    def test_geglu_clears_the_swiglu_flag(self):
        config = base_config()
        leg = dict(config["legs"][1], activation="geglu", protocol_version=6)
        worker = FakeWorker(6)
        worker.start()
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            wsm.drive_phone_leg(worker.port, config, leg, "sha256:" + "00" * 32, 4, Path(directory) / "L.jsonl",
                                clock=clock, sleep=clock.sleep)
        worker.join(5)
        self.assertEqual(struct.unpack_from("<H", worker.hello_bytes, 24)[0], wsm.F16_IO_FLAG)

    def test_shutdown_close_replaces_the_closing_call(self):
        config = base_config()
        leg = dict(config["legs"][1], protocol_version=6)
        worker = FakeWorker(6)
        worker.start()
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            sent = wsm.drive_phone_leg(worker.port, config, leg, "sha256:" + "00" * 32, 4,
                                       Path(directory) / "L.jsonl", clock=clock, sleep=clock.sleep, close="shutdown")
        worker.join(5)
        self.assertEqual(sent, wsm.phone_budget(config, leg) - 1)
        self.assertEqual(len(worker.executes), sent)
        self.assertEqual(worker.shutdown, (wsm.PROTOCOL_MAGIC, 6, 3, 0, -1, 0, 0, 0, 0, 0))

    def test_refused_hello_raises(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)

        def refuse():
            connection, _ = listener.accept()
            listener.close()
            with connection:
                wsm.receive(connection, 64)
                response = bytearray(96)
                struct.pack_into("<IHHH", response, 0, wsm.PROTOCOL_MAGIC, 6, 2, 7)
                connection.sendall(bytes(response))
        thread = threading.Thread(target=refuse, daemon=True)
        thread.start()
        config = base_config()
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "HELLO refused"):
            wsm.drive_phone_leg(listener.getsockname()[1], config, config["legs"][1], "sha256:" + "00" * 32, 4,
                                Path(directory) / "L.jsonl", clock=clock, sleep=clock.sleep)
        thread.join(5)


class ConfigTests(unittest.TestCase):
    def test_validation_and_plan(self):
        config = base_config()
        wsm.validate_config(config)
        plan = wsm.plan(config)
        self.assertEqual([leg["name"] for leg in plan["legs"]], ["cpu-ffn", "pixel-ffn"])
        self.assertEqual(plan["legs"][1]["budget_calls"], wsm.phone_budget(config, config["legs"][1]))
        for bad in (dict(base_config(), schema="x"),
                    base_config(legs=[dict(base_config()["legs"][1], op="attn_proj")]),
                    base_config(legs=[dict(base_config()["legs"][1], protocol_version=4)]),
                    base_config(legs=[dict(base_config()["legs"][1], worker={})])):
            with self.assertRaises(SystemExit):
                wsm.validate_config(bad)

    def test_desktop_command_matches_cadence(self):
        config = base_config()
        command = wsm.desktop_command(config, config["legs"][0], Path("/o/LEG.jsonl"))
        pairs = dict(zip(command[1::2], command[2::2]))
        self.assertEqual(pairs["--backend"], "CPU")
        self.assertEqual(pairs["--threads"], "8")
        self.assertEqual(pairs["--rows"], "1,2,4")
        self.assertEqual((pairs["--calls-per-token"], pairs["--gap-call-ms"], pairs["--gap-token-ms"]),
                         ("3", "5", "300"))
        self.assertEqual(pairs["--idle-before-s"], "5")

    def test_smoke_config_is_one_small_leg(self):
        small = wsm.smoke_config(base_config(), "pixel-ffn")
        self.assertEqual([leg["name"] for leg in small["legs"]], ["pixel-ffn"])
        self.assertEqual((small["rows"], small["idle_s"], small["cadence"]["warmup_tokens"]), ([1], 2, 1))
        with self.assertRaises(SystemExit):
            wsm.smoke_config(base_config(), "nope")

    def test_prepare_pins_worker_shard_and_libraries(self):
        config = base_config(legs=[{
            "name": "op15-qwen", "kind": "phone", "op": "ffn", "device": "op15", "transport": "adb-tcp",
            "protocol_version": 5, "activation": "swiglu", "layer": 0, "n_embd": 5120, "n_ff": 17408,
            "est_call_ms": {"1": 10, "2": 11, "4": 13},
            "worker": {"device_id": "op15-phone", "serial": "3C15", "adb_port": 5037, "worker_path": "/d/bin/w",
                       "library_directories": ["/d/bin"], "shard_path": "/d/s/HTP0.ffn.gguf",
                       "artifact_sha256": "sha256:" + "1" * 64, "column_quantum": 32, "max_tokens": 4,
                       "backend": "HTP0", "phone_port": 27001, "forward_port": 27001}}])
        listing = ("a" * 64 + "  /d/bin/w\n" + "b" * 64 + "  /d/s/HTP0.ffn.gguf\n" + "c" * 64 + "  /d/bin/libggml.so\n")

        class Completed:
            stdout = listing
        calls = []

        def run(argv, **kwargs):
            calls.append(argv)
            return Completed()
        with tempfile.TemporaryDirectory() as directory:
            pinned = wsm.prepare(config, Path(directory) / "P.json", run=run)
        pins = pinned["legs"][0]["worker"]["expected_sha256_by_path"]
        self.assertEqual(pins, {"/d/bin/w": "sha256:" + "a" * 64, "/d/s/HTP0.ffn.gguf": "sha256:" + "b" * 64,
                                "/d/bin/libggml.so": "sha256:" + "c" * 64})
        self.assertIn("su -c", calls[0][-1])
        worker, aoa = wsm.phone_worker_configuration(pinned, pinned["legs"][0])
        self.assertEqual((worker.backend, worker.layer_mask, worker.columns, worker.max_requests),
                         ("HTP0", 1, 17408, wsm.phone_budget(pinned, pinned["legs"][0])))
        self.assertIsNone(aoa)


def host_row(t_ns, cpu_energy_uj, gpu_mw):
    return {"gpu": {"sample_t_ns": t_ns, "power_mw": gpu_mw},
            "rapl_package": {"sample_t_ns": t_ns, "energy_uj": cpu_energy_uj, "max_energy_range_uj": 1 << 40}}


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def host_samples(self, profile):
        """profile: list of (t0_s, t1_s, cpu_w, gpu_w); samples every 0.2 s."""
        rows, energy, t = [], 0.0, 0.0
        end = profile[-1][1]
        while t <= end + 1e-9:
            cpu_w, gpu_w = next((c, g) for a, b, c, g in profile if a <= t < b or (t >= b and b == end))
            rows.append(host_row(int(t * 1e9), int(energy * 1e6), int(gpu_w * 1000)))
            energy += cpu_w * 0.2
            t = round(t + 0.2, 6)
        with (self.root / "HOST_SAMPLES.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    def write_leg(self, name, leg_config, records):
        directory = self.root / name
        directory.mkdir()
        (directory / "LEG_CONFIG.json").write_text(json.dumps(leg_config))
        with (directory / "LEG.jsonl").open("w") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")

    def test_desktop_incremental_energy_per_call(self):
        # idle 0-10 s at CPU 20 W / GPU 10 W; segment 10-20 s at 50 W / 10 W with 100 calls; idle 20-30 s
        self.host_samples([(0, 10, 20.0, 10.0), (10, 20, 50.0, 10.0), (20, 30, 20.0, 10.0)])
        calls = [{"kind": "call", "rows": 1, "warmup": False, "token": i, "position": 0,
                  "t_start_ns": int((10 + i * 0.1) * 1e9), "t_end_ns": int((10 + i * 0.1 + 0.02) * 1e9),
                  "compute_us": 20000.0} for i in range(100)]
        self.write_leg("cpu-ffn", {"name": "cpu-ffn", "kind": "desktop", "op": "ffn", "backend": "CPU"}, [
            {"kind": "meta", "logical_bytes": 534773760, "backend": "CPU"},
            {"kind": "idle", "phase": "before", "t0_ns": 0, "t1_ns": int(10e9)},
            *calls,
            {"kind": "segment", "rows": 1, "t0_ns": int(10e9), "t1_ns": int(20e9), "calls": 100},
            {"kind": "idle", "phase": "after", "t0_ns": int(20e9), "t1_ns": int(30e9)}])
        result = wsm.analyze(self.root)
        leg = result["legs"][0]
        self.assertAlmostEqual(leg["idle_w"]["cpu"], 20.0, places=1)
        segment = leg["segments"][0]
        self.assertAlmostEqual(segment["device_j_per_call"], 3.0, places=1)   # (50-20) W x 10 s / 100
        self.assertAlmostEqual(segment["device_gross_j_per_call"], 5.0, places=1)
        self.assertAlmostEqual(segment["system_j_per_call"], 3.0, places=1)   # GPU increment ~0
        self.assertAlmostEqual(segment["compute_ms_median"], 20.0)
        self.assertAlmostEqual(segment["bandwidth_gbs"], 534773760 / 0.02 / 1e9, places=3)
        self.assertAlmostEqual(segment["device_nj_per_byte"], 3.0 / 534773760 * 1e9, places=2)
        self.assertIsNone(segment["desktop_j_per_call"])  # a desktop leg has no separate waiting cost
        tex = wsm.latex(result)
        self.assertIn("cpu-ffn & 20.00", tex)
        self.assertIn("1.0$\\times$", tex)

    def test_phone_leg_uses_the_meter_window(self):
        self.host_samples([(0, 30, 15.0, 8.0)])
        # Pixel: phone uptime = host monotonic - 100 s; USB 5 V; 0.2 A idle (1 W), 0.6 A in the segment (3 W)
        log = []
        for i in range(0, 121):
            uptime = 100 + i * 0.25
            host = uptime - 100
            amps = 0.6 if 10 <= host < 20 else 0.2
            log.append(f"{uptime:.2f} usb/voltage_now=5000000 usb/current_now={int(amps * 1e6)} "
                       f"battery/charge_counter=4000000 battery/voltage_now=4000000 battery/current_now=0")
        (self.root / "pixel.txt").write_text("\n".join(log) + "\n")
        meter = {"pixel": {"serial": "5A040DLCH004ES", "local": str(self.root / "pixel.txt"),
                           "anchor_host_monotonic_s": 0.0, "anchor_phone_uptime_s": 100.0}}
        (self.root / "METER.json").write_text(json.dumps(meter))
        calls = [{"kind": "call", "rows": 1, "warmup": False, "token": i, "position": 0,
                  "t_start_ns": int((10 + i * 0.2) * 1e9), "t_end_ns": int((10 + i * 0.2 + 0.008) * 1e9),
                  "rpc_us": 8000.0, "compute_us": 7000} for i in range(50)]
        self.write_leg("pixel-ffn", {"name": "pixel-ffn", "kind": "phone", "op": "ffn", "device": "pixel"}, [
            {"kind": "meta", "logical_bytes": 534773760},
            {"kind": "idle", "phase": "before", "t0_ns": int(1e9), "t1_ns": int(9e9)},
            *calls,
            {"kind": "segment", "rows": 1, "t0_ns": int(10e9), "t1_ns": int(20e9), "calls": 50},
            {"kind": "idle", "phase": "after", "t0_ns": int(21e9), "t1_ns": int(29e9)}])
        result = wsm.analyze(self.root, self.root / "METER.json")
        segment = result["legs"][0]["segments"][0]
        self.assertAlmostEqual(result["legs"][0]["idle_w"]["pixel"], 1.0, places=2)
        self.assertAlmostEqual(segment["device_j_per_call"], (3.0 - 1.0) * 10 / 50, delta=0.02)  # 0.4 J (0.25 s samples)
        self.assertAlmostEqual(segment["compute_ms_median"], 7.0)
        self.assertAlmostEqual(segment["rpc_ms_median"], 8.0)
        self.assertAlmostEqual(segment["desktop_j_per_call"], 0.0, places=2)
        self.assertIn("pixel-ffn", wsm.markdown(result))

    def test_missing_measurements_print_tbd(self):
        result = {"legs": [{"name": "op15", "op": "ffn", "device": "op15", "idle_w": {}, "segments": [
            {"rows": 1, "calls": 10, "compute_ms_median": 9.3, "rpc_ms_median": None, "bandwidth_gbs": 57.5,
             "device_j_per_call": None, "desktop_j_per_call": None, "system_j_per_call": None,
             "device_nj_per_byte": None, "device_gross_j_per_call": None, "flags": []}]}]}
        tex = wsm.latex(result)
        self.assertIn("op15 & 9.30 & 57.5 & \\TBD", tex)
        self.assertIn("op15 & 9.30 & \\TBD & \\TBD", tex)


class RunTests(unittest.TestCase):
    def test_run_records_each_leg_and_the_host_samples(self):
        class Sampler:
            started = stopped = False

            def start(self):
                Sampler.started = True

            def stop(self):
                Sampler.stopped = True

            def rows(self):
                return [host_row(1, 0, 1000)]

            def diagnostics(self):
                return {"sample_count": 1}

        def desktop(config, leg, directory):
            (directory / "LEG.jsonl").write_text("")
            return {"exit_code": 0}

        def phone(config, leg, directory):
            raise RuntimeError("worker did not start")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            results = wsm.run(base_config(), output, sampler_factory=Sampler, desktop_runner=desktop,
                              phone_runner=phone, sleep=lambda s: None)
            self.assertEqual([(r["name"], r["status"]) for r in results],
                             [("cpu-ffn", "PASS"), ("pixel-ffn", "FAILED")])
            self.assertIn("worker did not start", results[1]["error"])
            self.assertTrue((output / "HOST_SAMPLES.jsonl").read_text().strip())
            self.assertEqual(json.loads((output / "HOST_SAMPLER_DIAGNOSTICS.json").read_text())["sample_count"], 1)
            self.assertTrue((output / "pixel-ffn" / "LEG_CONFIG.json").exists())
        self.assertTrue(Sampler.started and Sampler.stopped)

    def test_phone_leg_stops_the_session_with_the_calls_sent(self):
        config = base_config()
        leg = config["legs"][1]

        class Receipt:
            def __init__(self, value):
                self.value = value

            def to_json(self):
                return self.value

        class Contract:
            control_port = 0

        class Session:
            active = False
            stopped_with = None

            def preflight(self):
                return Receipt({"phase": "preflight"})

            def start(self, log):
                Session.active = True
                return Receipt({"phase": "launch"})

            def transport_contract(self):
                return Contract()

            def stop(self, served_calls):
                Session.stopped_with = served_calls
                Session.active = False
                return Receipt({"phase": "stop"})

        class Worker:
            artifact_sha256 = "sha256:" + "0" * 64
            max_tokens = 4
            serial, backend, worker_path, shard_path = "S", "CPU", "/w", "/s"

        worker = FakeWorker(6)
        worker.start()
        Contract.control_port = worker.port
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            receipts = wsm.run_phone_leg(config, leg, Path(directory), session_factory=lambda c, l: (Session(), Worker()),
                                         clock=clock, sleep=clock.sleep)
        worker.join(5)
        self.assertEqual(receipts["calls_sent"], wsm.phone_budget(config, leg))
        self.assertEqual(Session.stopped_with, receipts["calls_sent"])
        self.assertEqual(receipts["stop"], {"phase": "stop"})


FUNCTIONFS = {
    "session_script": "/data/local/tmp/probe/direct_phone_ffn_session.sh", "session_script_sha256": "sha256:" + "1" * 64,
    "restore_script": "/data/local/tmp/unified/restore_android_usb.sh", "restore_script_sha256": "sha256:" + "2" * 64,
    "session_root": "/data/local/tmp/ws16-ffs", "android_gadget_path": "/config/usb_gadget/g1",
    "functionfs_gadget_path": "/config/usb_gadget/g2", "functionfs_root_path": "/dev/usb-ffs/s41",
    "phone_usb_controller": "a600000.dwc3", "kernel_release": "6.12-ffs", "bridge_path": "",
    "bridge_sha256": "", "bridge_port": 27391, "usbfs_available_bytes": 134217728, "launch_timeout_s": 5,
    "usb_close_path": "/host/usb-close",
}


def functionfs_worker():
    from research_dev.scheduler.adapters.phone_tcp_session import AdbTcpWorkerConfiguration
    directory = "/data/local/tmp/bin"
    return AdbTcpWorkerConfiguration(
        device_id="op15-phone", serial="OP15", adb_port=5037, adb_path=Path("/usr/bin/adb"),
        worker_path=directory + "/llama-ffn-split-worker", library_directories=(directory,),
        shard_path="/data/local/tmp/ws16/HTP0.ffn.gguf", artifact_sha256="sha256:" + "a" * 64, layer_mask=1,
        n_embd=8, columns=64, column_quantum=64, max_tokens=4, swiglu=True, backend="HTP0", phone_port=27391,
        max_requests=0, worker_environment={"ADSP_LIBRARY_PATH": directory, "GGML_HEXAGON_NHVX": "4",
                                            "GGML_HEXAGON_VMEM": "3328", "S41_DISABLE_GRAPH_CACHE": "1"},
        expected_sha256_by_path={directory + "/llama-ffn-split-worker": "sha256:" + "3" * 64,
                                 "/data/local/tmp/ws16/HTP0.ffn.gguf": "sha256:" + "4" * 64})


class FakeBridge:
    """A bridge process that, once waited for, wrote its terminal receipt."""

    def __init__(self, log, calls, returncode=0):
        self.log, self.calls, self.returncode_value, self.pid = Path(log.name), calls, returncode, 4242
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            with self.log.open("a") as log:  # the session closed its handle after the launch
                log.write("[ffn-dmabuf-bridge] ready\nFFNDMABUF " + json.dumps(
                    {"status": "ok", "calls": self.calls, "allocator": "devmem", "reset_recoveries": 0}) + "\n")
            self.returncode = self.returncode_value
        return self.returncode

    def terminate(self):
        self.returncode = -15

    kill = terminate


class FunctionFsSessionTests(unittest.TestCase):
    def make(self, directory, *, calls=10, gadget=None, terminal_status="0", fail_launch=False):
        from research_dev.scheduler.adapters.bridge import AndroidUsbRestorationReceipt, FunctionFsUsbObservation
        bridge = Path(directory) / "bridge-bin"
        bridge.write_bytes(b"bridge")
        functionfs = dict(FUNCTIONFS, bridge_path=str(bridge),
                          bridge_sha256="sha256:" + __import__("hashlib").sha256(b"bridge").hexdigest())
        worker = functionfs_worker()
        state = {"gadget": gadget if gadget is not None else [False, True], "commands": [], "listening": False,
                 "closed": 0}
        pins = dict(worker.expected_sha256_by_path, **{functionfs["session_script"]: functionfs["session_script_sha256"],
                                                       functionfs["restore_script"]: functionfs["restore_script_sha256"]})

        class Done:
            def __init__(self, stdout="", returncode=0):
                self.stdout, self.returncode, self.stderr = stdout, returncode, ""

        def run(command, **_):
            if command[0] == "/host/usb-close":
                state["closed"] += 1
                state["gadget"] = [False]
                return Done("FFNUSB_CLOSE status=ok")
            remote = command[-1]
            state["commands"].append(remote)
            if "uname -r" in remote:
                return Done("6.12-ffs\n")
            if "sha256sum" in remote:
                return Done("".join(f"{digest[7:]}  {path}\n" for path, digest in pins.items()))
            if "boot_id" in remote:
                return Done("boot-1\n")
            if "ps -A -o PID,PPID,ARGS" in remote:  # the script's watchdog subshell outlives its cleanup
                root = next((c.split("mkdir -p ")[1].split(" ")[0] for c in state["commands"] if "nohup env" in c), "")
                return Done("  PID  PPID ARGS\n  700     1 sh " + functionfs["session_script"] + " "
                            + worker.worker_path + " x 0 64 HTP0 " + root + " r 1800 0 a\n  701   700 sleep 1800\n"
                            + "  702     1 sh /x/run.sh " + worker.worker_path + "\n" if state.get("ended") else "")
            if "nohup env" in remote and fail_launch:
                raise subprocess.TimeoutExpired(command, 30)
            if "terminal.status" in remote and "cat" in remote:
                return Done(terminal_status + "\n" if state.get("ended") else "")
            if "worker.log" in remote:
                return Done("[ffn-worker] DMA-BUF complete\n")
            return Done("")

        def probe():
            up = state["gadget"].pop(0) if len(state["gadget"]) > 1 else state["gadget"][0]
            if not up:
                raise RuntimeError("FunctionFS USB device observation is not unique")
            return FunctionFsUsbObservation("2-9.1", "18d1", "2d00", 10000, "host")

        def popen(command, stdout, **_):
            state["bridge_command"] = command
            state["listening"] = True
            state["bridge"] = FakeBridge(stdout, calls)
            return state["bridge"]

        def restored(timeout_s):
            state["ended"] = True
            return AndroidUsbRestorationReceipt("OP15", 5037, "2-9.1", "22d9", "2580", 10000)

        clock = Clock()
        session = wsm.FunctionFsBridgePhoneWorkerSession(
            worker, functionfs, run=run, popen=popen, sleep=clock.sleep, clock=lambda: clock() / 1e9,
            probe=probe, restored=restored, listening=lambda port: state["listening"])
        return session, state

    def test_start_runs_the_production_session_script_without_a_budget_and_stop_acknowledges(self):
        with tempfile.TemporaryDirectory() as directory:
            session, state = self.make(directory, gadget=[False, False, True, True, False])
            self.assertEqual(session.preflight().to_json()["phase"], "preflight")
            launch = session.start(Path(directory) / "worker.log").to_json()
            self.assertTrue(session.active)
            self.assertEqual(session.transport_contract().control_port, 27391)
            remote = next(c for c in state["commands"] if "nohup env" in c)
            for word in ("SCHEDULER_FUNCTIONFS_GADGET=/config/usb_gadget/g2", "S41_FFN_F16_IO=1",
                         "S41_FFN_MAX_TOKENS=4", "S41_FFN_COLUMN_QUANTUM=64", "S41_FFN_QUEUE_DEPTH=1",
                         "S41_FFN_VMEM=3328", "/data/local/tmp/ws16/HTP0.ffn.gguf 0 64 HTP0",
                         "/data/local/tmp/unified/restore_android_usb.sh 1800 0 sha256:"):
                self.assertIn(word, remote)
            self.assertEqual(state["bridge_command"][1:], ["127.0.0.1", "27391", "devmem", "1", "134217728",
                                                           "functionfs-dmabuf-async-ring-v2"])
            self.assertEqual(launch["usb"]["product_id"], "2d00")
            stop = session.stop(served_calls=10).to_json()
            self.assertFalse(session.active)
            self.assertEqual((stop["bridge_returncode"], stop["phone_terminal_status"]), (0, "0"))
            self.assertTrue(any("host-ack" in c for c in state["commands"]))
            self.assertIn("DMA-BUF complete", (Path(directory) / "phone-worker.log").read_text())
            self.assertEqual(state["closed"], 0)
            self.assertEqual(stop["orphans_killed"], [700, 701])
            self.assertTrue(any(c.startswith("su -c 'kill -KILL 700 701") or "kill -KILL 700 701" in c
                                for c in state["commands"]))

    def test_stop_refuses_a_call_count_mismatch_and_closes_a_gadget_left_up(self):
        with tempfile.TemporaryDirectory() as directory:
            session, state = self.make(directory, calls=9, gadget=[True])
            session.preflight = lambda: None
            session._boot_id = "boot-1"
            session.start(Path(directory) / "worker.log")
            with self.assertRaisesRegex(RuntimeError, "served 9 calls, the driver sent 10"):
                session.stop(served_calls=10)
            self.assertEqual(state["closed"], 1)
            self.assertFalse(session.active)

    def test_failed_launch_restores_android_usb(self):
        with tempfile.TemporaryDirectory() as directory:
            session, state = self.make(directory, fail_launch=True, gadget=[False])
            session._boot_id = "boot-1"
            with self.assertRaises(subprocess.TimeoutExpired):
                session.start(Path(directory) / "worker.log")
            self.assertFalse(session.active)
            self.assertTrue(any("restore_android_usb.sh" in c and "test -f" in c for c in state["commands"]))
            self.assertTrue(any("host-ack" in c for c in state["commands"]))

    def test_preflight_ignores_processes_that_only_name_the_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            session, state = self.make(directory)
            state["ended"] = True  # the ps fake lists a watchdog subshell and a script naming the worker
            self.assertEqual(session.preflight().to_json()["phase"], "preflight")

    def test_environment_the_script_does_not_reproduce_is_refused(self):
        import dataclasses
        worker = dataclasses.replace(functionfs_worker(), worker_environment={"GGML_HEXAGON_MBUF": "2048"})
        with self.assertRaisesRegex(ValueError, "fixed environment"):
            wsm.FunctionFsBridgePhoneWorkerSession(worker, FUNCTIONFS)

    def test_config_requires_the_functionfs_section(self):
        leg = {"name": "op15-ffs", "kind": "phone", "op": "ffn", "device": "op15", "transport": "functionfs-bridge",
               "protocol_version": 6, "activation": "swiglu", "layer": 0, "n_embd": 8, "n_ff": 64,
               "est_call_ms": {"1": 8.0, "2": 9.0, "4": 10.0}, "worker": {}}
        with self.assertRaisesRegex(SystemExit, "functionfs needs session_script"):
            wsm.validate_config(base_config(legs=[leg]))
        wsm.validate_config(base_config(legs=[dict(leg, functionfs=FUNCTIONFS)]))


if __name__ == "__main__":
    unittest.main()
