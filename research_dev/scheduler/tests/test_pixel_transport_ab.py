"""WS10 Phase-B A/B harness mechanics on this host (no phone): every arm kind and transport end to end.

adb is faked (phone commands run locally; ``adb forward`` is a real local TCP proxy), the FFN worker is the build-cpu
worker on a tiny model, the relay is the host build and a SOCK_SEQPACKET "cable" that keeps the latest connection
per side stands in for the accessory link. Checks the ABBA order, finite-budget stops, byte identity across
transports, both receipts and the analysis table; numbers mean nothing here.
"""

from __future__ import annotations

import json
from pathlib import Path
import select
import shutil
import socket
import stat
import struct
import sys
import tempfile
import threading
import unittest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from research_dev.scheduler.adapters import aoa_bridge as ab  # noqa: E402
from research_dev.scheduler.adapters.phone_aoa_session import file_sha256  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import pixel_transport_ab as tool  # noqa: E402
from test_aoa_bridge import FakeSysfs, build_host_relay  # noqa: E402
from test_aoa_bridge_session import FakeUsb, free_port  # noqa: E402
from test_two_phone_helpers import BIN_DIR, PIXEL_SERIAL  # noqa: E402

FAKE_ADB = r'''#!/usr/bin/env python3
"""adb stand-in: phone commands run on this host; forwards are local TCP proxies (state in $FAKE_ADB_STATE)."""
import os, signal, subprocess, sys
state = os.environ["FAKE_ADB_STATE"]
args, serial = sys.argv[1:], "?"
while args and args[0] in ("-P", "-s"):
    if args[0] == "-s":
        serial = args[1]
    args = args[2:]
if args[:2] == ["shell", "-T"]:
    os.execvp("sh", ["sh", "-c", args[2]])
if args[:1] == ["shell"]:
    command = args[1].replace("ps -A -o PID,ARGS", "ps -A -o pid,args")
    sys.exit(subprocess.call(["sh", "-c", command]))
if args[:1] == ["get-state"]:
    print("device")
    sys.exit(0)
if args[:2] == ["forward", "--no-rebind"]:
    host, phone = args[2].split(":")[1], args[3].split(":")[1]
    proxy = subprocess.Popen([sys.executable, os.path.join(state, "proxy.py"), host, phone],
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    proxy.stdout.readline()
    with open(os.path.join(state, "forward-" + host), "w") as out:
        out.write(f"{proxy.pid} {serial} {phone}\n")
    print(host)
    sys.exit(0)
if args[:2] == ["forward", "--remove"]:
    path = os.path.join(state, "forward-" + args[2].split(":")[1])
    pid = int(open(path).read().split()[0])
    os.kill(pid, signal.SIGTERM)
    os.remove(path)
    sys.exit(0)
if args[:2] == ["forward", "--list"]:
    for name in os.listdir(state):
        if name.startswith("forward-"):
            pid, owner, phone = open(os.path.join(state, name)).read().split()
            print(f"{owner} tcp:{name[8:]} tcp:{phone}")
    sys.exit(0)
sys.exit(2)
'''

PROXY = r'''import socket, sys, threading
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", int(sys.argv[1])))
listener.listen(8)
print("ready", flush=True)
def pump(a, b):
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    for end in (a, b):
        try:
            end.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
while True:
    client, _ = listener.accept()
    try:
        upstream = socket.create_connection(("127.0.0.1", int(sys.argv[2])))
    except OSError:
        client.close()
        continue
    for a, b in ((client, upstream), (upstream, client)):
        threading.Thread(target=pump, args=(a, b), daemon=True).start()
'''


class ReconnectingCable:
    """SOCK_SEQPACKET host/phone endpoints; the newest connection on each side is the live one and messages to
    an absent peer are dropped (a bridge restart does not disturb the relay, as on real USB)."""

    def __init__(self, root: Path) -> None:
        self.paths = (str(root / "cable-host.sock"), str(root / "cable-phone.sock"))
        self.ends = [None, None]
        self.lock = threading.Lock()
        self.listeners = []
        for side, path in enumerate(self.paths):
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            listener.bind(path)
            listener.listen(4)
            self.listeners.append(listener)
            threading.Thread(target=self._accept, args=(side, listener), daemon=True).start()

    def _accept(self, side: int, listener: socket.socket) -> None:
        while True:
            try:
                connection, _ = listener.accept()
            except OSError:
                return
            with self.lock:
                self.ends[side] = connection
            threading.Thread(target=self._pump, args=(side, connection), daemon=True).start()

    def _pump(self, side: int, connection: socket.socket) -> None:
        while True:
            try:
                select.select([connection], [], [])
                data = connection.recv(1 << 18)
            except (OSError, ValueError):
                data = b""
            if not data:
                with self.lock:
                    if self.ends[side] is connection:
                        self.ends[side] = None
                return
            with self.lock:
                peer = self.ends[1 - side]
            if peer is not None:
                try:
                    peer.sendall(data)
                except OSError:
                    pass

    def close(self) -> None:
        for listener in self.listeners:
            listener.close()


class PixelTransportAbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        worker = BIN_DIR / "llama-ffn-split-worker"
        if not worker.exists():
            raise unittest.SkipTest(f"llama-ffn-split-worker missing under {BIN_DIR}")
        if shutil.which("flock") is None:
            raise unittest.SkipTest("flock missing")
        from tiny_llama_gguf import N_EMBD, N_FF, write_tiny_llama_gguf
        cls.directory = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls.directory.name)
        cls.relay = build_host_relay(root)
        cls.model = write_tiny_llama_gguf(root / "tiny.gguf", n_layer=4)
        cls.worker = root / "bin" / "llama-ffn-split-worker"
        cls.worker.parent.mkdir()
        shutil.copy2(worker, cls.worker)
        state = root / "adb-state"
        state.mkdir()
        (state / "proxy.py").write_text(PROXY)
        cls.adb = root / "adb"
        cls.adb.write_text(FAKE_ADB.replace('os.environ["FAKE_ADB_STATE"]', repr(str(state))))
        cls.adb.chmod(cls.adb.stat().st_mode | stat.S_IEXEC)
        inputs = root / "inputs"
        inputs.mkdir()
        for layer in (2, 3):
            for row in range(4):
                values = [((layer * 13 + row * 5 + index) % 89 - 44) / 64 for index in range(N_EMBD)]
                (inputs / f"input-layer{layer}-row{row}.f16").write_bytes(struct.pack(f"<{N_EMBD}e", *values))
        cls.sysfs = FakeSysfs(root / "sysfs")
        cls.sysfs.set("2-9.2", PIXEL_SERIAL, "18d1", "2d01", 47)
        cls.cable = ReconnectingCable(root)
        cls.config = {
            "schema": tool.SCHEMA, "input_dir": str(inputs), "forward_port": free_port(),
            "echo_phone_port": free_port(), "echo_host_port": free_port(), "repeats": 2, "cooldown_s": 0,
            "cadence": {"calls_per_token": 6, "gap_call_ms": 1.0, "gap_token_ms": [2.0, 4.0], "seed": 3},
            "segments": [{"name": "m1", "rows": 1, "warmup_tokens": 1, "tokens": 2},
                         {"name": "m2", "rows": 2, "warmup_tokens": 1, "tokens": 2},
                         {"name": "m4", "rows": 4, "warmup_tokens": 1, "tokens": 2}],
            "phone": {"serial": PIXEL_SERIAL, "adb_path": str(cls.adb), "worker_path": str(cls.worker),
                      "worker_sha256": file_sha256(cls.worker), "library_directories": [str(BIN_DIR.resolve())],
                      "shard_path": str(cls.model), "artifact_sha256": file_sha256(cls.model),
                      "common_sha256": {str(cls.model): file_sha256(cls.model)}, "layers": [2, 3],
                      "n_embd": N_EMBD, "columns": N_FF, "column_quantum": 64, "max_tokens": 4,
                      "phone_port": free_port(), "worker_environment": {"S42_TEST": "1"}, "as_root": False,
                      "launch_timeout_s": 60.0},
            "aoa_bridge": {"usb_sysfs_device": "2-9.2", "relay_path": str(cls.relay),
                           "relay_sha256": file_sha256(cls.relay),
                           "bridge_script_sha256": file_sha256(Path(ab.__file__)),
                           "relay_lock_path": str(root / "relay.lock"), "python_path": sys.executable,
                           "bridge_options": {"status_interval_ms": 200}},
            "arms": [
                {"name": "adb", "transport": "adb-tcp"},
                {"name": "aoa", "transport": "aoa-bridge"},
                {"name": "aoa-ka", "transport": "aoa-bridge", "trace": True,
                 "bridge_options": {"keepalive_ms": 2, "keepalive_window_ms": 100},
                 "relay_options": {"qos_window_ms": 200}},
                {"name": "adb-echo", "kind": "echo", "transport": "adb-tcp"},
                {"name": "aoa-echo", "kind": "echo", "transport": "aoa-bridge"},
                {"name": "aoa-local", "kind": "echo", "transport": "aoa-local"},
                {"name": "life", "kind": "lifecycle", "transport": "aoa-bridge"},
            ],
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.cable.close()
        cls.directory.cleanup()

    def harness(self, usb: FakeUsb) -> tool.Harness:
        return tool.Harness(self.config, sleep=lambda seconds: None, sysfs_root=self.sysfs.root,
                            session_options={"observe_usb": usb.observe, "switch_mode": usb.switch},
                            relay_extra=("--accessory-unix", self.cable.paths[1]),
                            bridge_extra=("--link-unix", self.cable.paths[0]))

    def test_plan_orders_abba_and_estimates(self) -> None:
        order = [arm["run"] for arm in tool.arm_order(self.config)]
        self.assertEqual(order[:6], ["01-adb", "02-aoa", "03-aoa-ka", "04-adb-echo", "05-aoa-echo", "06-aoa-local"])
        self.assertEqual(order[6:], ["07-aoa-local", "08-aoa-echo", "09-adb-echo", "10-aoa-ka", "11-aoa", "12-adb",
                                     "13-life"])
        self.assertEqual(tool.calls_per_arm(self.config), 3 * 3 * 6)
        self.assertGreater(tool.estimate_seconds(self.config), 0)

    def test_every_arm_runs_stops_cleanly_and_outputs_are_byte_identical(self) -> None:
        usb = FakeUsb(product="4ee7")
        output = self.root / "ab"
        results = tool.run(self.config, output, self.harness(usb))
        self.assertEqual(usb.switches, 1)  # the first arm put the Pixel in accessory mode for every arm
        self.assertEqual([row["status"] for row in results], ["PASS"] * 13)
        for row in results:
            if row["kind"] == "ffn":
                self.assertEqual(row["calls"], tool.calls_per_arm(self.config))
                self.assertEqual((row["stop"]["details"] if "details" in row["stop"] else row["stop"])["exit_code"], 0)
        summary = tool.analyze(output)
        self.assertEqual(summary["byte_identity"], "PASS")
        identity = json.loads((output / "aoa-bridge-byte-identity.json").read_text())
        self.assertEqual(identity["distinct_output_hashes"], 3)  # one per segment, shared by every ffn arm
        trip = json.loads((output / "aoa-bridge-round-trip.json").read_text())
        self.assertEqual(trip["calibration_arm"], "aoa")
        self.assertEqual(set(trip["overhead_us_by_rows"]), {"1", "2", "4"})
        table = (output / "AB_TABLE.md").read_text()
        self.assertIn("| 01-adb | adb | adb-tcp | 1 |", table)
        traced = [row for row in summary["table"] if row.get("arm") == "aoa-ka" and row.get("trace")]
        self.assertTrue(traced and traced[0]["trace"]["phone_residence_us"]["n"] > 0)
        echo = json.loads((output / "06-aoa-local" / "RESULT.json").read_text())
        self.assertEqual(echo["endpoint_stop"]["bridge_exit_code"], 0)
        lifecycle = json.loads((output / "aoa-bridge-scheduler-launched-session.json").read_text())
        self.assertEqual(lifecycle["status"], "PASS")
        self.assertFalse(lifecycle["steps"]["alive_after_drop"])
        self.assertEqual(lifecycle["steps"]["release_lost"]["aoa_bridge"]["stop"]["relay_pids_after"], [])
        keepalive = json.loads((output / "03-aoa-ka" / "RESULT.json").read_text())
        counters = keepalive["stop"]["aoa_bridge"]["stop"]["bridge_final_status"]["counters"]
        self.assertGreater(counters.get("nops", 0), 0)


if __name__ == "__main__":
    unittest.main()
