"""WS6 link-latency sweep: the host-side TCP delay proxy on loopback and its opt-in rig wiring.

The proxy tests measure added round trip against a loopback echo server (the Pixel helper's adb
forward stands in); the wiring tests check that ``helper_phones[].link_delay_proxy_port`` makes
llama-server dial the proxy while the worker session keeps its adb forward, and that without the
key every declaration is unchanged.
"""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from research_dev.scheduler.campaigns.burstgpt import catalog as campaign_catalog
from research_dev.scheduler.campaigns.burstgpt.tools.link_delay_proxy import (
    DirectionConfig, ProxyConfig, ProxyThread, parse_config,
)
from research_dev.scheduler._internal.plan_contracts.common import RuntimePlanError
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler.configuration.common import SchedulerConfigurationError
from research_dev.scheduler.configuration.rig import RigManifest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import two_phone_harness as h  # noqa: E402
from test_two_phone_helpers import two_phone_rig_json  # noqa: E402

REPO_ROOT = TESTS_DIR.parents[2]
CALL_BYTES = 10240  # one Qwen3-14B FFN row (n_embd 5120, f16)


class EchoServer:
    """Blocking loopback echo server, one thread per connection (the phone worker's role)."""

    def __init__(self) -> None:
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen()
        self.port = self.socket.getsockname()[1]
        self.eof_seen = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                connection, _ = self.socket.accept()
            except OSError:
                return
            threading.Thread(target=self._echo, args=(connection,), daemon=True).start()

    def _echo(self, connection: socket.socket) -> None:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with connection:
            while data := connection.recv(1 << 16):
                connection.sendall(data)
            self.eof_seen.set()

    def close(self) -> None:
        self.socket.close()


def _connect(port: int) -> socket.socket:
    client = socket.create_connection(("127.0.0.1", port), timeout=10)
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return client


def _receive(client: socket.socket, size: int) -> bytes:
    chunks, got = [], 0
    while got < size:
        chunk = client.recv(size - got)
        if not chunk:
            raise ConnectionError("closed")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def median_rtt_ms(port: int, calls: int = 30) -> float:
    """Median round trip of synchronous 10 KB calls with a 2 ms gap (the server's call pattern)."""
    samples = []
    payload = os.urandom(CALL_BYTES)
    with _connect(port) as client:
        for _ in range(calls):
            start = time.perf_counter()
            client.sendall(payload)
            if _receive(client, CALL_BYTES) != payload:
                raise AssertionError("echo corrupted")
            samples.append(time.perf_counter() - start)
            time.sleep(0.002)
    return statistics.median(samples) * 1e3


def proxy(port: int, up_ms: float = 0.0, down_ms: float = 0.0, up_jitter_ms: float = 0.0,
          down_jitter_ms: float = 0.0) -> ProxyThread:
    return ProxyThread(ProxyConfig("127.0.0.1", 0, "127.0.0.1", port,
                                   DirectionConfig(up_ms / 1e3, up_jitter_ms / 1e3),
                                   DirectionConfig(down_ms / 1e3, down_jitter_ms / 1e3)))


class LinkDelayProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.echo = EchoServer()
        self.addCleanup(self.echo.close)

    def test_added_round_trip_matches_the_configured_delay(self) -> None:
        direct = median_rtt_ms(self.echo.port)
        with proxy(self.echo.port) as zero:
            through_zero = median_rtt_ms(zero.port)
        with proxy(self.echo.port, 2.5, 2.5) as delayed:
            through = median_rtt_ms(delayed.port)
            status = delayed.proxy.status()
        # the 0-delay proxy hop is small; the delayed proxy adds its configured 5 ms round trip on top
        self.assertLess(through_zero - direct, 2.0)
        self.assertGreaterEqual(through - direct, 5.0 - 0.3)
        self.assertLessEqual(through - through_zero, 5.0 + 2.0)
        for name in ("up", "down"):
            applied = status["directions"][name]["applied_delay"]
            self.assertAlmostEqual(applied["p50_ms"], 2.5, delta=1.0)
            self.assertEqual(status["directions"][name]["bytes"], 30 * CALL_BYTES)
        self.assertEqual(status["config"]["up"], {"delay_ms": 2.5, "jitter_ms": 0.0})

    def test_each_direction_has_its_own_delay(self) -> None:
        with proxy(self.echo.port) as zero:
            base = median_rtt_ms(zero.port)
        with proxy(self.echo.port, up_ms=4.0) as up_only:
            added = median_rtt_ms(up_only.port) - base
            status = up_only.proxy.status()
        self.assertGreaterEqual(added, 4.0 - 0.3)
        self.assertLessEqual(added, 4.0 + 2.0)
        self.assertLess(status["directions"]["down"]["applied_delay"]["p50_ms"], 1.0)

    def test_jitter_keeps_bytes_and_order(self) -> None:
        messages = [f"{index:06d}|".encode() * 50 for index in range(300)]
        sent = b"".join(messages)
        with proxy(self.echo.port, 1.0, 1.0, 3.0, 3.0) as jittered:
            with _connect(jittered.port) as client:
                sender = threading.Thread(target=lambda: [client.sendall(row) for row in messages])
                sender.start()
                received = _receive(client, len(sent))
                sender.join()
        self.assertEqual(received, sent)

    def test_large_transfer_and_end_of_stream_pass_through(self) -> None:
        payload = os.urandom(3 * 1024 * 1024)
        with proxy(self.echo.port, 0.5, 0.5) as delayed:
            with _connect(delayed.port) as client:
                sender = threading.Thread(target=client.sendall, args=(payload,))
                sender.start()
                received = _receive(client, len(payload))
                sender.join()
                client.shutdown(socket.SHUT_WR)
                self.assertTrue(self.echo.eof_seen.wait(5))
                self.assertEqual(client.recv(1), b"")
            deadline = time.monotonic() + 5
            while delayed.proxy.connections["closed"] < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(delayed.proxy.connections, {"accepted": 1, "closed": 1, "upstream_failures": 0,
                                                          "errors": 0})
        self.assertEqual(received, payload)

    def test_a_refused_upstream_closes_the_client(self) -> None:
        closed = socket.socket()
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
        closed.close()
        with proxy(port, 1.0, 1.0) as lonely:
            with _connect(lonely.port) as client:
                try:
                    self.assertEqual(client.recv(1), b"")
                except ConnectionResetError:
                    pass
            deadline = time.monotonic() + 5
            while lonely.proxy.connections["upstream_failures"] < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(lonely.proxy.connections["upstream_failures"], 1)

    def test_command_line_writes_ready_and_status_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ready, status = Path(directory) / "READY.json", Path(directory) / "STATUS.json"
            process = subprocess.Popen(
                [sys.executable, "-m", "research_dev.scheduler.campaigns.burstgpt.tools.link_delay_proxy",
                 "--listen", "127.0.0.1:0", "--upstream", f"127.0.0.1:{self.echo.port}", "--rtt-ms", "3",
                 "--ready-file", str(ready), "--status", str(status), "--status-interval-s", "0.2"],
                cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 20
                while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
                    time.sleep(0.02)
                announced = json.loads(ready.read_text())
                self.assertEqual(announced["config"]["up"]["delay_ms"], 1.5)
                self.assertGreater(median_rtt_ms(announced["port"], calls=10), 3.0 - 0.3)
            finally:
                process.send_signal(signal.SIGTERM)
                output, error = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, error)
            self.assertIn("LINK_DELAY_PROXY_READY", output)
            final = json.loads(status.read_text())
            self.assertTrue(final["stopped"])
            self.assertEqual(final["directions"]["up"]["bytes"], 10 * CALL_BYTES)
            self.assertEqual(final["connections"]["accepted"], 1)

    def test_round_trip_splits_evenly_and_excludes_one_way_flags(self) -> None:
        config, _ = parse_config(["--listen", "127.0.0.1:26992", "--upstream", "127.0.0.1:26991", "--rtt-ms", "7"])
        self.assertEqual((config.up.delay_s, config.down.delay_s, config.listen_port, config.upstream_port),
                         (0.0035, 0.0035, 26992, 26991))
        config, _ = parse_config(["--listen", "127.0.0.1:1", "--upstream", "127.0.0.1:2", "--up-delay-ms", "2",
                                  "--down-jitter-ms", "1"])
        self.assertEqual((config.up.delay_s, config.down.delay_s, config.down.jitter_s), (0.002, 0.0, 0.001))
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            parse_config(["--listen", "127.0.0.1:1", "--upstream", "127.0.0.1:2", "--rtt-ms", "1",
                          "--up-delay-ms", "1"])
        with self.assertRaises(ValueError):
            DirectionConfig(-0.001)


class LinkDelayWiringTests(unittest.TestCase):
    """``link_delay_proxy_port``: opt-in; the server dials the proxy, the worker keeps its adb forward."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = h.manifest(self.directory.name)
        self.index = Path(self.directory.name) / "FFN_SHARDS.json"
        self.index.write_text(json.dumps({"parent_sha256": self.model.artifact_sha256,
                                          "schema": "s42-ffn-shard-index-v1", "shards": [{
            "columns": 128, "layer_mask": "%016x" % h.PIXEL_MASK, "n_ff": 128,
            "parent_sha256": self.model.artifact_sha256, "path": "HTP0.ffn.gguf", "session_id": "HTP0",
            "shard_bytes": 4096, "shard_sha256": h.SHARD_SHA, "weight_type": "F16"}]}), encoding="ascii")
        self.config = SimpleNamespace(
            helper_phone_ffn_shards={"pixel10pro-phone": (self.index, "/data/local/tmp/pixel-shards")},
            phone_ffn_shard_index_path=None, phone_ffn_shard_directory=None)

    def _rig_json(self, proxy_port: int | None) -> dict:
        row = two_phone_rig_json()
        row["helper_phones"][0]["forward_port"] = 26991  # the campaign's fixed adb forward
        if proxy_port is not None:
            row["helper_phones"][0]["link_delay_proxy_port"] = proxy_port
        return row

    def test_absent_key_round_trips_and_keeps_the_declaration(self) -> None:
        row = self._rig_json(None)
        rig = RigManifest.from_json(row, Path("/"))
        self.assertEqual(rig.helper_phones[0].link_delay_proxy_port, 0)
        self.assertNotIn("link_delay_proxy_port", rig.helper_phones[0].to_json())
        self.assertEqual(json.dumps(rig.to_json(), sort_keys=True), json.dumps(row, sort_keys=True))
        helper, = campaign_catalog.helper_phone_co_helpers(rig, self.config, self.model).helpers
        self.assertEqual(dict(helper.transport_parameters)["ffn_worker_port"], 26991)
        self.assertNotIn("ffn_link_proxy_upstream_port", helper.transport_parameters)

    def test_proxy_port_is_dialled_and_the_forward_is_kept_upstream(self) -> None:
        row = self._rig_json(26992)
        rig = RigManifest.from_json(row, Path("/"))
        self.assertEqual(rig.helper_phones[0].to_json()["link_delay_proxy_port"], 26992)
        self.assertEqual(json.dumps(rig.to_json(), sort_keys=True), json.dumps(row, sort_keys=True))
        declaration = campaign_catalog.helper_phone_co_helpers(rig, self.config, self.model)
        helper, = declaration.helpers
        self.assertEqual(dict(helper.transport_parameters), {
            "adb_port": 5037, "adb_serial": h.PIXEL_SERIAL, "ffn_transport": "adb-tcp",
            "ffn_worker_host": "127.0.0.1", "ffn_worker_port": 26992, "ffn_link_proxy_upstream_port": 26991,
            "phone_worker_port": 26990})
        from research_dev.scheduler.adapters.co_helper_lifecycle import CoHelperLifecycle
        from research_dev.scheduler.adapters.phone_helpers import PhoneHelperBinding, helper_server_environment
        from research_dev.scheduler.adapters.phone_transport import phone_transport_contract

        # the adb-forward worker session (forward_port 26991) matches this declaration
        worker = SimpleNamespace(configuration=SimpleNamespace(
            device_id=helper.device_id, serial=helper.serial, layer_mask=helper.layer_mask,
            column_quantum=helper.column_quantum, max_tokens=helper.max_tokens, phone_port=26990,
            forward_port=26991))
        CoHelperLifecycle(declaration, {helper.device_id: worker})
        worker.configuration.forward_port = 26992
        with self.assertRaises(PhysicalAdapterError):
            CoHelperLifecycle(declaration, {helper.device_id: worker})
        # llama-server's helper environment dials the proxy
        contract = phone_transport_contract(helper.transport_parameters)
        self.assertEqual((contract.control_host, contract.control_port), ("127.0.0.1", 26992))
        op15 = PhoneHelperBinding(device_id="op15-phone", serial=h.OP15_SERIAL,
                                  layer_mask=1 << h.PIXEL_MASK.bit_length(), label="op15")
        environment = helper_server_environment(
            (op15, PhoneHelperBinding(device_id=helper.device_id, serial=helper.serial,
                                              layer_mask=helper.layer_mask, label="pixel",
                                              transport_parameters=dict(helper.transport_parameters))),
            {"S41_SERVER_FFN_LAYER_MASK": str(op15.layer_mask | h.PIXEL_MASK)},
            (phone_transport_contract(dict(h.FUNCTIONFS_PARAMETERS)), contract))
        self.assertEqual(environment["S41_SERVER_FFN_HELPER1_PORT"], "26992")

    def test_the_live_forward_check_compares_the_upstream_port(self) -> None:
        rig = RigManifest.from_json(self._rig_json(26992), Path("/"))
        declaration = campaign_catalog.helper_phone_co_helpers(rig, self.config, self.model)
        helper, = declaration.helpers
        from research_dev.scheduler.adapters.co_helper_lifecycle import CoHelperLifecycle

        class Worker:
            def __init__(self, live_port: int) -> None:
                self.configuration = SimpleNamespace(
                    device_id=helper.device_id, serial=helper.serial, layer_mask=helper.layer_mask,
                    column_quantum=helper.column_quantum, max_tokens=helper.max_tokens, phone_port=26990,
                    forward_port=26991)
                self.live_port = live_port

            def preflight(self):
                return SimpleNamespace(to_json=lambda: {"kind": "preflight"})

            def start(self, log_path):
                return SimpleNamespace(to_json=lambda: {"kind": "start"})

            def transport_parameters(self):
                return {**h.PIXEL_TRANSPORT, "ffn_worker_port": self.live_port}

        CoHelperLifecycle(declaration, {helper.device_id: Worker(26991)})._start(helper, Path("/w.log"))
        with self.assertRaisesRegex(PhysicalAdapterError, "forward differs"):
            CoHelperLifecycle(declaration, {helper.device_id: Worker(26992)})._start(helper, Path("/w.log"))

    def test_proxy_port_must_be_its_own_port(self) -> None:
        for port, message in ((26991, "ambiguous"), (29382, "ambiguous")):
            with self.subTest(port=port), self.assertRaisesRegex(SchedulerConfigurationError, message):
                RigManifest.from_json(self._rig_json(port), Path("/"))
        row = self._rig_json(26992)
        row["helper_phones"][0]["forward_port"] = 0
        with self.assertRaisesRegex(SchedulerConfigurationError, "fixed forward port"):
            RigManifest.from_json(row, Path("/"))
        from research_dev.scheduler._internal.plan_contracts.co_helpers import RuntimeCoHelperPhone
        helper = h.co_helpers().helpers[0]
        with self.assertRaisesRegex(RuntimePlanError, "own port"):
            RuntimeCoHelperPhone(
                device_id=helper.device_id, serial=helper.serial, label=helper.label, session_id=helper.session_id,
                layer_mask=helper.layer_mask, column_quantum=helper.column_quantum, max_tokens=helper.max_tokens,
                shard_sha256=helper.shard_sha256, resident_bytes=helper.resident_bytes,
                transport_parameters={**h.PIXEL_TRANSPORT, "ffn_link_proxy_upstream_port": 26991})


if __name__ == "__main__":
    unittest.main()
