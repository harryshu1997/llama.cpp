"""WS10: AoaBridgePhoneWorkerSession end to end on this host.

The real CPU FFN worker (build-cpu) runs behind a fake adb (phone commands run locally), the phone relay is the
host build of native/aoa_bridge/s43_aoa_relay.c and the host bridge is adapters/aoa_bridge.py as a subprocess; a
SOCK_SEQPACKET "cable" stands in for the accessory bulk endpoints, and USB mode switching is faked. Checks the
lifecycle (preflight, switch, launch, liveness, loss, finite and resident stops), byte identity against a direct
worker connection, and the refusals.
"""

from __future__ import annotations

from pathlib import Path
import select
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from research_dev.scheduler.adapters import aoa_bridge as ab  # noqa: E402
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError  # noqa: E402
from research_dev.scheduler.adapters.phone_aoa_session import (  # noqa: E402
    AoaBridgeConfiguration, AoaBridgePhoneWorkerSession, file_sha256,
)
from research_dev.scheduler.adapters.phone_tcp_session import (  # noqa: E402
    EXECUTE_REQUEST, EXECUTE_RESPONSE, HELLO_REQUEST, HELLO_RESPONSE, PROTOCOL_MAGIC, PROTOCOL_VERSION,
    AdbTcpWorkerConfiguration, _fnv32,
)
from test_aoa_bridge import build_host_relay  # noqa: E402
from test_two_phone_helpers import BIN_DIR, PIXEL_SERIAL  # noqa: E402

FAKE_ADB = r'''#!/usr/bin/env python3
"""adb stand-in: runs phone shell commands on this host; the device is always online."""
import os, subprocess, sys
args = sys.argv[1:]
while args and args[0] in ("-P", "-s"):
    args = args[2:]
if args[:2] == ["shell", "-T"]:
    os.execvp("sh", ["sh", "-c", args[2]])
if args[:1] == ["shell"]:
    command = args[1].replace("ps -A -o PID,ARGS", "ps -A -o pid,args")
    sys.exit(subprocess.call(["sh", "-c", command]))
if args[:1] == ["get-state"]:
    print("device")
    sys.exit(0)
sys.exit(2)
'''


class Cable:
    """Two SOCK_SEQPACKET endpoints (host, phone); every message crosses unchanged, like a USB transfer."""

    def __init__(self, root: Path) -> None:
        self.host_path, self.phone_path = str(root / "cable-host.sock"), str(root / "cable-phone.sock")
        self.listeners = []
        for path in (self.host_path, self.phone_path):
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            listener.bind(path)
            listener.listen(1)
            self.listeners.append(listener)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        ends = [listener.accept()[0] for listener in self.listeners]
        while True:
            ready, _, _ = select.select(ends, [], [])
            for index in (0, 1):
                if ends[index] in ready:
                    try:
                        data = ends[index].recv(1 << 18)
                    except OSError:
                        data = b""
                    if not data:
                        for end in ends:
                            try:
                                end.shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass
                        return
                    ends[1 - index].sendall(data)

    def close(self) -> None:
        for listener in self.listeners:
            listener.close()


class FakeUsb:
    def __init__(self, product: str = "4ee7", serial: str = PIXEL_SERIAL) -> None:
        self.product, self.serial = product, serial
        self.switches = 0

    def observe(self, device: str) -> ab.UsbDeviceState:
        return ab.UsbDeviceState(device, self.serial, "18d1", self.product, 2, 47, 5000)

    def switch(self, device, serial, *, forbidden_serials, timeout_s):
        self.switches += 1
        before = self.observe(device).to_json()
        self.product = "2d01"
        return {"already_accessory": False, "before": before, "after": self.observe(device).to_json()}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class AoaBridgeSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.worker = BIN_DIR / "llama-ffn-split-worker"
        if not cls.worker.exists():
            raise unittest.SkipTest(f"llama-ffn-split-worker missing under {BIN_DIR}")
        if shutil.which("flock") is None:
            raise unittest.SkipTest("flock missing")
        from tiny_llama_gguf import N_EMBD, N_FF, write_tiny_llama_gguf
        cls.n_embd, cls.n_ff = N_EMBD, N_FF
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.relay = build_host_relay(cls.root)
        cls.model = write_tiny_llama_gguf(cls.root / "tiny.gguf", n_layer=4)
        cls.artifact = file_sha256(cls.model)
        cls.adb = cls.root / "adb"
        cls.adb.write_text(FAKE_ADB)
        cls.adb.chmod(cls.adb.stat().st_mode | stat.S_IEXEC)
        cls.worker_copy = cls.root / "bin" / "llama-ffn-split-worker"
        cls.worker_copy.parent.mkdir()
        shutil.copy2(cls.worker, cls.worker_copy)
        cls.count = 0

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def setUp(self) -> None:
        type(self).count += 1
        self.case = self.root / f"case{self.count}"
        self.case.mkdir()
        self.cable = Cable(self.case)
        self.addCleanup(self.cable.close)
        self.usb = FakeUsb()

    def configurations(self, **overrides):
        worker = dict(
            device_id="pixel10pro-phone", serial=PIXEL_SERIAL, adb_port=5037, adb_path=self.adb,
            worker_path=str(self.worker_copy), library_directories=(str(BIN_DIR.resolve()),),
            shard_path=str(self.model), artifact_sha256=self.artifact, layer_mask=0b1100, n_embd=self.n_embd,
            columns=self.n_ff, column_quantum=64, max_tokens=4, swiglu=True, backend="CPU", phone_port=free_port(),
            forward_port=free_port(), max_requests=7, worker_environment={"S42_TEST": "1"},
            expected_sha256_by_path={str(self.worker_copy): file_sha256(self.worker_copy),
                                     str(self.model): self.artifact},
            launch_timeout_s=60.0)
        bridge = dict(usb_sysfs_device="2-9.2", relay_path=str(self.relay), relay_sha256=file_sha256(self.relay),
                      bridge_script_sha256=file_sha256(Path(ab.__file__)), relay_lock_path=str(self.case / "relay.lock"),
                      python_path=sys.executable, bridge_options={"status_interval_ms": 200},
                      relay_options={"qos_window_ms": 500}, ready_timeout_s=60)
        for key, value in overrides.items():
            (bridge if key in bridge or key in AoaBridgeConfiguration.__dataclass_fields__ else worker)[key] = value
        return AdbTcpWorkerConfiguration(**worker), AoaBridgeConfiguration(**bridge)

    def session(self, **overrides) -> AoaBridgePhoneWorkerSession:
        worker, bridge = self.configurations(**overrides)
        session = AoaBridgePhoneWorkerSession(
            worker, bridge, observe_usb=self.usb.observe, switch_mode=self.usb.switch,
            extra_bridge_arguments=("--link-unix", self.cable.host_path),
            extra_relay_arguments=("--accessory-unix", self.cable.phone_path))
        self.addCleanup(self._cleanup, session)
        return session

    @staticmethod
    def _cleanup(session) -> None:
        for process in (session._bridge_process, session._relay_process, session._process):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        for path in (session.bridge.relay_path, session.configuration.worker_path):
            subprocess.run(["pkill", "-TERM", "-f", "^" + path], check=False)

    def hello(self, stream, configuration) -> None:
        stream.sendall(HELLO_REQUEST.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, 1, configuration.layer_mask,
                                          configuration.n_embd, configuration.columns, 1 | 2, configuration.max_tokens,
                                          bytes.fromhex(configuration.artifact_sha256[7:])))
        hello = HELLO_RESPONSE.unpack(AoaBridgePhoneWorkerSession._receive(stream, HELLO_RESPONSE.size))
        self.assertEqual(hello[:4], (PROTOCOL_MAGIC, PROTOCOL_VERSION, 2, 0))

    def calls(self, port: int, configuration, payloads, first_id: int = 1) -> list[bytes]:
        outputs = []
        with socket.create_connection(("127.0.0.1", port), timeout=60) as stream:
            stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.hello(stream, configuration)
            for offset, (layer, rows, payload) in enumerate(payloads):
                request_id = first_id + offset
                stream.sendall(EXECUTE_REQUEST.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, 3, request_id, layer,
                                                    configuration.n_embd * rows, len(payload), _fnv32(payload),
                                                    configuration.columns, rows) + payload)
                response = EXECUTE_RESPONSE.unpack(AoaBridgePhoneWorkerSession._receive(stream,
                                                                                        EXECUTE_RESPONSE.size))
                output = AoaBridgePhoneWorkerSession._receive(stream, len(payload))
                self.assertEqual(response[:6], (PROTOCOL_MAGIC, PROTOCOL_VERSION, 4, 0, 0, request_id))
                self.assertEqual(response[9], _fnv32(output))
                outputs.append(output)
        return outputs

    def wait_disconnected(self, session) -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            text = session._log_path.read_text()
            if text.rfind("client disconnected") >= text.rfind("client connected"):
                return
            time.sleep(0.05)
        self.fail("the worker still has a client")

    def payloads(self):
        """Finite F16 activations (the worker refuses non-finite inputs), rows 1, 2 and 4."""
        rows = []
        for index, (layer, count) in enumerate(((2, 1), (3, 2), (2, 4))):
            values = [((index * 7 + value) % 97 - 48) / 64 for value in range(self.n_embd * count)]
            rows.append((layer, count, struct.pack("<%de" % len(values), *values)))
        return rows

    def test_switch_launch_bridge_and_finite_budget_stop(self) -> None:
        session = self.session(max_requests=9)
        preflight = session.preflight()
        self.assertEqual(preflight.details["aoa_bridge"]["usb"]["vendor_product"], "18d1:4ee7")
        launch = session.start(self.case / "worker.log")
        self.assertEqual(self.usb.switches, 1)
        details = launch.details["aoa_bridge"]
        self.assertTrue(details["mode"]["switched"])
        self.assertEqual(len(details["relay_pids"]), 1)
        self.assertEqual(launch.details["host_endpoint"], "aoa-bridge")
        parameters = session.transport_parameters()
        self.assertEqual((parameters["ffn_worker_port"], parameters["ffn_link_transport"], parameters["ffn_transport"]),
                         (session.configuration.forward_port, "aoa-bridge", "adb-tcp"))
        self.assertEqual(session.transport_contract().control_port, session.configuration.forward_port)
        self.assertTrue(session.alive())
        self.assertTrue(session.alive(connect=True))
        payloads = self.payloads()
        through_bridge = self.calls(session.configuration.forward_port, session.configuration, payloads)
        self.wait_disconnected(session)
        direct = self.calls(session.configuration.phone_port, session.configuration, payloads, first_id=10)
        self.assertEqual(through_bridge, direct)  # byte identity: the relay/bridge never touch the bytes
        self.wait_disconnected(session)
        stop = session.stop(served_calls=6)
        self.assertEqual((stop.details["exit_code"], stop.details["drained_calls"], stop.details["signalled"]),
                         (0, 3, False))
        record = stop.details["aoa_bridge"]["stop"]
        self.assertEqual((record["bridge_exit_code"], record["relay_client_exit_code"], record["relay_pids_after"]),
                         (0, 0, []))
        self.assertEqual(record["bridge_final_status"]["state"], "down")
        self.assertTrue(stop.details["forward_removed"])
        self.assertFalse(session.active)

    def test_already_accessory_resident_worker_stops_only_when_idle(self) -> None:
        self.usb.product = "2d01"
        session = self.session(max_requests=0)
        launch = session.start(self.case / "worker.log")
        self.assertFalse(launch.details["aoa_bridge"]["mode"]["switched"])
        self.assertEqual(self.usb.switches, 0)
        self.calls(session.configuration.forward_port, session.configuration, self.payloads()[:1])
        self.wait_disconnected(session)
        with self.assertRaisesRegex(PhysicalAdapterError, "stays running"):
            session.stop()
        stop = session.stop(allow_idle_signal=True)
        self.assertTrue(stop.details["signalled"])
        self.assertEqual(stop.details["worker_pids_after"], [])
        self.assertEqual(stop.details["aoa_bridge"]["stop"]["relay_pids_after"], [])

    def test_a_dead_bridge_is_a_lost_helper_and_release_clears_the_phone(self) -> None:
        session = self.session(max_requests=0)
        session.start(self.case / "worker.log")
        self.assertFalse(session.client_exited())
        session._bridge_process.send_signal(signal.SIGKILL)
        session._bridge_process.wait(timeout=10)
        self.assertFalse(session.alive())
        self.assertTrue(session.client_exited())
        release = session.release_lost()
        self.assertEqual(release.details["aoa_bridge"]["stop"]["relay_pids_after"], [])
        self.assertEqual(session._worker_pids(), [])
        self.assertFalse(session.active)

    def test_preflight_refusals(self) -> None:
        cases = {
            "relay differs": {"relay_sha256": "sha256:" + "0" * 64},
            "host bridge script differs": {"bridge_script_sha256": "sha256:" + "0" * 64},
        }
        for message, change in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(PhysicalAdapterError, message):
                self.session(**change).preflight()
        session = self.session()
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", session.configuration.forward_port))
            busy.listen(1)
            with self.assertRaisesRegex(PhysicalAdapterError, "is occupied"):
                session.preflight()
        self.assertIsNone(session._boot_id)
        for product, serial, message in (("2d00", PIXEL_SERIAL, "accessory without adb"),
                                         ("2d00", "SCHEDFFN0001", "forbidden phone"),
                                         ("4ee7", "OTHER", "not the pinned Pixel")):
            with self.subTest(product=product, serial=serial):
                self.usb.product, self.usb.serial = product, serial
                with self.assertRaisesRegex(PhysicalAdapterError, message):
                    self.session().preflight()
        self.usb.product, self.usb.serial = "4ee7", PIXEL_SERIAL

    def test_configuration_validation_and_argv(self) -> None:
        worker, bridge = self.configurations()
        self.assertEqual(bridge.relay_argv(7074)[:3], (str(self.relay), "--worker-port", "7074"))
        self.assertIn("--qos-window-ms", bridge.relay_argv(7074))
        argv = bridge.bridge_argv(Path("/b.py"), PIXEL_SERIAL, 26991, Path("/r"), Path("/s"), None)
        self.assertEqual(argv[:3], (sys.executable, "/b.py", "serve"))
        self.assertEqual(argv.count("--forbid-serial"), 2)
        self.assertEqual(AoaBridgeConfiguration.from_json(bridge.to_json()), bridge)
        self.assertEqual(AoaBridgeConfiguration.from_json(bridge.to_json()).options_sha256, bridge.options_sha256)
        for change in ({"relay_options": {"qos_latency_us": -1}}, {"relay_options": {"spin": 1}},
                       {"relay_options": {"cpus": "0"}}, {"bridge_options": {"keepalive_ms": 5000}},
                       {"usb_sysfs_device": "2-9..2"}, {"relay_sha256": "abc"}, {"python_path": "python3"},
                       {"relay_options": {"qos_window_ms": 1.5}}):
            with self.subTest(change=change), self.assertRaises(PhysicalAdapterError):
                AoaBridgeConfiguration(**{**bridge.to_json(), "forbidden_serials": bridge.forbidden_serials, **change})
        with self.assertRaisesRegex(PhysicalAdapterError, "unknown|invalid"):
            AoaBridgeConfiguration.from_json({**bridge.to_json(), "surprise": 1})
        from dataclasses import replace
        with self.assertRaisesRegex(PhysicalAdapterError, "fixed host port"):
            AoaBridgePhoneWorkerSession(replace(worker, forward_port=0), bridge)
        with self.assertRaisesRegex(PhysicalAdapterError, "forbidden"):
            AoaBridgePhoneWorkerSession(worker, replace(bridge, forbidden_serials=(PIXEL_SERIAL,)))
        rooted = AoaBridgePhoneWorkerSession(replace(worker, as_root=True), bridge)
        command = rooted.relay_shell_command()
        self.assertTrue(command.startswith("su -c "))
        self.assertIn("flock -n 9", command)


if __name__ == "__main__":
    unittest.main()
