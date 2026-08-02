#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
S39 = HERE.parents[2]
sys.path.insert(0, str(S39))

import direct_relay_selftest as relay_test


def load_parser():
    source = HERE.parent / "producers_v1" / "direct_frame_evidence_v1.py"
    spec = importlib.util.spec_from_file_location(
        "s39_direct_frame_relay_parser",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load direct frame parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frames = load_parser()


class DirectFrameRelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("S39_RELAY_BINARY")
        cls.binary = (
            Path(configured)
            if configured
            else REPO / "build-cuda" / "bin" / "llama-stage-direct-relay"
        )
        if not cls.binary.is_file():
            raise unittest.SkipTest("llama-stage-direct-relay is not built")

    def run_one(self, emit_frames):
        head = relay_test.MockWorker(0, 30, False)
        tail = relay_test.MockWorker(30, 40, True)
        head.start()
        tail.start()
        relay_port = relay_test.free_port()
        argv = [
            str(self.binary),
            "--listen",
            str(relay_port),
            "--head",
            f"127.0.0.1:{head.port}",
            "--tail",
            f"127.0.0.1:{tail.port}",
        ]
        if emit_frames:
            argv.append("--emit-direct-frames")
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        client = relay_test.connect_with_retry(relay_port)
        client.hello()
        rows = (
            relay_test.BatchRow(10, 1, 0, 0, 7),
            relay_test.BatchRow(11, 1, 1, 0, 9),
        )
        client.batch(rows)
        client.remove(0, 10, 1)
        client.remove(1, 11, 1)
        client.drain()
        client.stop()
        client.close()
        stdout, stderr = process.communicate(timeout=5)
        head.join()
        tail.join()
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(stdout, b"")
        return stderr

    def test_opt_in_frame_hash_and_lineage_match_transmitted_bytes(self):
        stderr = self.run_one(True)
        payload = struct.pack(
            "<8f",
            7.0,
            8.0,
            9.0,
            10.0,
            9.0,
            10.0,
            11.0,
            12.0,
        )
        expected = [{
            "hidden_width": 4,
            "positions": [0, 0],
            "request_ids": [10, 11],
            "route_epochs": [1, 1],
            "rows": 2,
            "seq_ids": [0, 1],
        }]
        records = frames.parse_direct_frames(stderr, expected, [payload])
        self.assertEqual(records[0]["activation_payload_bytes"], 32)

    def test_default_relay_emits_no_frame_records(self):
        stderr = self.run_one(False)
        self.assertNotIn(frames.PREFIX, stderr)


if __name__ == "__main__":
    unittest.main()
