"""WS13: the hardware fault injector against the Pixel's AOA (direct USB) transport, end to end on this host.

Same rig stand-ins as test_aoa_bridge_session: the real CPU FFN worker behind a fake adb (phone commands run
locally; here ``su -c`` is unwrapped as on the rooted Pixel), the host build of the phone relay, the real host
bridge as a subprocess and a SOCK_SEQPACKET "cable" -- one that can be re-plugged, because a re-join starts a new
relay and a new bridge. The launch receipt goes into CO_HELPER_LIFECYCLE.json exactly as CoHelperLifecycle writes
it for elastic phones (receipt + ``monotonic_ns``). Then:

* inject_helper_loss.py finds the AOA worker from that receipt (path, adb port, pids) and SIGTERMs it;
* a server-style helper connection that was open through the bridge ends instead of hanging (the relay sees the
  worker's socket close and sends CLOSE; the bridge closes the server's socket): llama-server's helper error;
* the session reports the loss the way the rig's liveness monitor reads it (client exited, not alive), its
  release clears worker, relay and bridge, and a re-join on the same session object serves byte-identical calls.
"""

from __future__ import annotations

import json
from pathlib import Path
import select
import socket
import sys
import threading
import time
import unittest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from research_dev.scheduler.adapters.phone_tcp_session import (  # noqa: E402
    EXECUTE_REQUEST, PROTOCOL_MAGIC, PROTOCOL_VERSION, _fnv32,
)
from research_dev.scheduler.campaigns.burstgpt.tools import inject_helper_loss as injector  # noqa: E402
import test_aoa_bridge_session as aoa_session_tests  # noqa: E402  (module import: its TestCase is not re-collected)

SU_FAKE_ADB = r'''#!/usr/bin/env python3
"""adb stand-in: runs phone shell commands on this host; ``su -c CMD`` runs CMD (the rooted Pixel)."""
import os, shlex, subprocess, sys
args = sys.argv[1:]
while args and args[0] in ("-P", "-s"):
    args = args[2:]
if args[:2] == ["shell", "-T"]:
    os.execvp("sh", ["sh", "-c", args[2]])
if args[:1] == ["shell"]:
    command = args[1]
    if command.startswith("su -c "):
        command = shlex.split(command)[2]
    command = command.replace("ps -A -o PID,ARGS", "ps -A -o pid,args")
    sys.exit(subprocess.call(["sh", "-c", command]))
if args[:1] == ["get-state"]:
    print("device")
    sys.exit(0)
sys.exit(2)
'''

G11_ELASTIC_PHONES = {"drop_recovery": True, "helper_loss_recovery": "mask_out", "join": True,
                      "join_probe_interval_s": 10, "max_readmissions_per_device": 3, "readmission_cooldown_s": 60}


class ReusableCable:
    """The test_aoa_bridge_session cable, re-plugged after each disconnect (a re-join's relay and bridge)."""

    def __init__(self, root: Path) -> None:
        self.host_path, self.phone_path = str(root / "cable-host.sock"), str(root / "cable-phone.sock")
        self.listeners = []
        for path in (self.host_path, self.phone_path):
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            listener.bind(path)
            listener.listen(1)
            self.listeners.append(listener)
        self.plugs = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            try:
                ends = [listener.accept()[0] for listener in self.listeners]
            except OSError:
                return
            self.plugs += 1
            self._pump(ends)

    @staticmethod
    def _pump(ends) -> None:
        while True:
            ready, _, _ = select.select(ends, [], [])
            for index in (0, 1):
                if ends[index] not in ready:
                    continue
                try:
                    data = ends[index].recv(1 << 18)
                except OSError:
                    data = b""
                if not data:
                    for end in ends:
                        for action in (lambda: end.shutdown(socket.SHUT_RDWR), end.close):
                            try:
                                action()
                            except OSError:
                                pass
                    return
                ends[1 - index].sendall(data)

    def close(self) -> None:
        for listener in self.listeners:
            listener.close()


class AoaFaultInjectionTests(aoa_session_tests.AoaBridgeSessionTests):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.adb.write_text(SU_FAKE_ADB)

    def setUp(self) -> None:
        type(self).count += 1
        self.case = self.root / f"fault{self.count}"
        self.case.mkdir()
        self.cable = ReusableCable(self.case)
        self.addCleanup(self.cable.close)
        self.usb = aoa_session_tests.FakeUsb()

    def run_dir(self) -> Path:
        run_dir = self.case / "campaign" / "run"
        run_dir.mkdir(parents=True)
        (run_dir.parent / injector.RESOLVED_CONFIGURATION_FILE).write_text(
            json.dumps({"campaign": {"elastic_phones": G11_ELASTIC_PHONES}}))
        return run_dir

    def test_injector_terminates_the_aoa_worker_the_server_sees_the_loss_and_the_helper_rejoins(self) -> None:
        session = self.session(max_requests=0)   # elastic co-helpers are resident workers
        configuration, port = session.configuration, session.configuration.forward_port
        run_dir = self.run_dir()
        receipts = [session.preflight().to_json()]
        # CoHelperLifecycle._start(elastic=True): the launch receipt plus the monotonic anchor of ``t=``
        receipts.append({**session.start(self.case / "worker.log", release_failed_start=True).to_json(),
                         "monotonic_ns": time.monotonic_ns()})
        (run_dir / injector.LIFECYCLE_FILE).write_text(json.dumps(receipts))
        launch = injector.latest_launch(run_dir, configuration.device_id)
        self.assertEqual(launch["host_endpoint"], "aoa-bridge")
        self.assertEqual((injector.launched_worker_path(launch), injector.launched_adb_port(launch)),
                         (configuration.worker_path, configuration.adb_port))
        self.assertEqual(injector.launched_layer_range(launch), (2, 3))   # ``active=`` counts these layers' calls
        payloads = self.payloads()
        before = self.calls(port, configuration, payloads)
        self.wait_disconnected(session)

        # a server-style helper connection is open through the bridge when the worker is terminated
        stream = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.addCleanup(stream.close)
        self.hello(stream, configuration)
        record = injector.inject(injector.parse_args([
            "--device", configuration.device_id, "--when", "t=0", "--signal", "TERM", "--run-dir", str(run_dir),
            "--authorized", "--adb", str(self.adb), "--timeout-s", "20", "--poll-s", "0.05"]))
        self.assertEqual(record["status"], "INJECTED")
        self.assertEqual(record["signalled_pids"], launch["worker_pids"])
        self.assertTrue((run_dir / injector.RESULT_FILE).is_file())

        layer, rows, payload = payloads[0]
        stream.settimeout(20)
        try:  # the next call cannot hang: the connection ends (EOF or reset) = the server's helper error
            stream.sendall(EXECUTE_REQUEST.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, 3, 50, layer,
                                                configuration.n_embd * rows, len(payload), _fnv32(payload),
                                                configuration.columns, rows) + payload)
            ended = stream.recv(64) == b""
        except (ConnectionResetError, BrokenPipeError):
            ended = True
        self.assertTrue(ended)

        deadline = time.monotonic() + 20
        while not session.client_exited() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(session.client_exited())   # what the rig's liveness monitor reads
        self.assertFalse(session.alive())
        release = session.release_lost()
        self.assertEqual(release.details["aoa_bridge"]["stop"]["relay_pids_after"], [])
        self.assertTrue(release.details["boot_unchanged"])
        self.assertEqual(session._worker_pids(), [])
        self.assertFalse(session.active)

        # join: the lifecycle's elastic start again on the same session object (fresh worker, relay, bridge)
        session.preflight()
        rejoin = session.start(self.case / "worker-join.log", release_failed_start=True)
        self.assertNotEqual(rejoin.details["worker_pids"], launch["worker_pids"])
        self.assertEqual(self.cable.plugs, 2)
        after = self.calls(port, configuration, payloads, first_id=20)
        self.assertEqual(after, before)            # byte identity across the loss
        self.wait_disconnected(session)
        stop = session.stop(allow_idle_signal=True)
        self.assertEqual(stop.details["aoa_bridge"]["stop"]["relay_pids_after"], [])

    def test_a_static_run_or_a_foreign_process_is_never_signalled(self) -> None:
        session = self.session(max_requests=0)
        run_dir = self.case / "static" / "run"
        run_dir.mkdir(parents=True)
        (run_dir.parent / injector.RESOLVED_CONFIGURATION_FILE).write_text(json.dumps({"campaign": {}}))
        with self.assertRaisesRegex(injector.InjectionError, "does not declare elastic_phones"):
            injector.inject(injector.parse_args([
                "--device", session.configuration.device_id, "--when", "t=0", "--signal", "TERM",
                "--run-dir", str(run_dir), "--authorized", "--adb", str(self.adb), "--timeout-s", "1"]))
        run_dir = self.run_dir()
        receipts = [{**session.start(self.case / "worker.log", release_failed_start=True).to_json(),
                     "monotonic_ns": time.monotonic_ns()}]
        receipts[0]["worker_pids"] = [1]           # a pid that does not run the launched worker
        (run_dir / injector.LIFECYCLE_FILE).write_text(json.dumps(receipts))
        record = injector.inject(injector.parse_args([
            "--device", session.configuration.device_id, "--when", "t=0", "--signal", "TERM",
            "--run-dir", str(run_dir), "--authorized", "--adb", str(self.adb), "--timeout-s", "5"]))
        self.assertEqual((record["status"], record["signalled_pids"]), ("NOT_INJECTED", []))
        self.assertTrue(session.alive())
        session.stop(allow_idle_signal=True)


# the base class's own tests run in their module; here only the fault-injection tests
for _name in [name for name in vars(aoa_session_tests.AoaBridgeSessionTests) if name.startswith("test")]:
    setattr(AoaFaultInjectionTests, _name, None)


if __name__ == "__main__":
    unittest.main()
