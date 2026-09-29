"""The fault tool's ``active=SECONDS`` trigger fires only while the helper is serving calls.

Hardware lesson (G1/G1c, 2026-09-25): ``request=NNN`` fires once a request has COMPLETED (streams are
written at request end) and ``t=SECONDS`` may land between two helper phases, so both hit an idle helper.
``active`` watches the newest hot-model server's per-call ``S41SERVERFFNCALL ... layer=L`` lines for the
helper's own layers and fires the moment their count grows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from research_dev.scheduler.campaigns.burstgpt.tools import inject_helper_loss as tool


LAUNCH_COMMAND = [
    "/usr/bin/adb", "-P", "5037", "-s", "SERIAL1", "shell", "-T",
    "su -c 'exec 9>/data/local/tmp/.lock\nflock -n 9 9>&9 || exit 73\n"
    "exec env S42_X=1 /data/local/tmp/w/llama-ffn-split-worker -m /data/local/tmp/w/QWEN.ffn.gguf "
    "--layers 18-23 --columns 17408 --port 26990 --bind 127.0.0.1'",
]


def _call_line(layer: int) -> str:
    return (f"S41SERVERFFNCALL context=6162:2:1:1 request=7 layer={layer} tokens=1 columns=17408 "
            "payload_bytes=10240\n")


class ActiveTriggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)
        (self.run_dir / tool.LIFECYCLE_FILE).write_text(json.dumps([
            {"phase": "preflight", "device_id": "pixel10pro-phone", "serial": "SERIAL1"},
            {"phase": "launch", "device_id": "pixel10pro-phone", "serial": "SERIAL1", "worker_pids": [21569],
             "monotonic_ns": 1_000_000_000, "command": LAUNCH_COMMAND},
        ]))
        self.stderr = self.run_dir / "large-model-5-physical-hot-desktop.stderr"
        self.stderr.write_text("S41SERVERFFN ready host=usb\n" + _call_line(0) + _call_line(17))
        self.state = {"device_id": "pixel10pro-phone"}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _met(self, seconds: float = 300.0, now_s: float = 1000.0):
        return tool.condition_met(self.run_dir, ("active", seconds), min_stream_bytes=1,
                                  monotonic=lambda: now_s, state=self.state)

    def test_parse_accepts_active_seconds_and_rejects_negatives(self) -> None:
        self.assertEqual(tool.parse_condition("active=300"), ("active", 300.0))
        with self.assertRaises(argparse.ArgumentTypeError):
            tool.parse_condition("active=-1")
        with self.assertRaises(argparse.ArgumentTypeError):
            tool.parse_condition("active")

    def test_layer_range_comes_from_the_launch_receipt(self) -> None:
        launch = tool.latest_launch(self.run_dir, "pixel10pro-phone")
        self.assertEqual(tool.launched_layer_range(launch), (18, 23))
        self.assertIsNone(tool.launched_layer_range({"command": ["adb", "shell", "worker --port 1"]}))

    def test_fires_only_when_the_helper_layers_are_being_called(self) -> None:
        # Before SECONDS: never, whatever the log says.
        self.assertIsNone(self._met(now_s=1.0 + 100.0))
        # First observation only records the baseline (host layers 0 and 17 do not count).
        self.assertIsNone(self._met())
        self.assertEqual(self.state["helper_calls"], 0)
        # Host-layer calls keep arriving: still idle for the helper.
        with self.stderr.open("a") as out:
            out.write(_call_line(3))
        self.assertIsNone(self._met())
        # A call on one of the helper's layers: serving now -> fire, with the evidence recorded.
        with self.stderr.open("a") as out:
            out.write(_call_line(19) + _call_line(20))
        met = self._met()
        self.assertIsNotNone(met)
        self.assertEqual((met["kind"], met["layers"], met["helper_calls_before"], met["helper_calls_after"]),
                         ("active", [18, 23], 0, 2))
        self.assertEqual(met["server_stderr"], str(self.stderr))

    def test_uses_the_newest_hot_server_log(self) -> None:
        self.assertIsNone(self._met())
        newer = self.run_dir / "large-model-14-physical-hot-desktop.stderr"
        newer.write_text(_call_line(21))
        import os
        os.utime(self.stderr, (1, 1))
        met = self._met()
        self.assertIsNotNone(met)
        self.assertEqual(met["server_stderr"], str(newer))

    def test_without_state_or_layers_the_trigger_stays_closed(self) -> None:
        self.assertIsNone(tool.condition_met(self.run_dir, ("active", 0.0), min_stream_bytes=1,
                                             monotonic=lambda: 1000.0, state=None))
        (self.run_dir / tool.LIFECYCLE_FILE).write_text(json.dumps([
            {"phase": "launch", "device_id": "pixel10pro-phone", "serial": "SERIAL1", "worker_pids": [1],
             "monotonic_ns": 1_000_000_000, "command": ["adb", "shell", "worker --port 1"]}]))
        with self.stderr.open("a") as out:
            out.write(_call_line(19))
        self.assertIsNone(self._met())


if __name__ == "__main__":
    unittest.main()
