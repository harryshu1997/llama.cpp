#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
LIVE = HERE.parent
S15 = LIVE.parent / "s15_runtime_dispatch"
S14 = LIVE.parent / "s14_mixed_streaming_scheduler"
sys.path[:0] = [str(LIVE), str(S15), str(S14)]

from executor_contract import BOUNDARY_SCHEMA, ExecutionRequest  # noqa: E402
from prepared_transport import PreparedSubprocessTransport  # noqa: E402


def request(timeout_us=1_000_000) -> ExecutionRequest:
    return ExecutionRequest(
        1, "r", "sha256:" + "11" * 32, "d", 1, 1, 1, 1, 1, "k", ("x",),
        "sha256:" + "22" * 32, "sha256:" + "33" * 32,
        timeout_us, BOUNDARY_SCHEMA,
    )


class PreparedTransportTests(unittest.TestCase):
    def test_setup_is_outside_exchange_elapsed(self) -> None:
        code = (
            "import sys,time; time.sleep(.15); "
            "sys.stderr.write('LAUNCHER_READY {\\\"device_boot_id\\\":\\\"b\\\"}\\n'); "
            "sys.stderr.flush(); sys.stdin.buffer.read(); sys.stdout.write('{}\\n')"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            transport = PreparedSubprocessTransport((sys.executable, "-c", code), path, 2)
            reply = transport.exchange(request(), b"{}\n")
            self.assertEqual(reply.state, "reply")
            self.assertLess(reply.elapsed_us, 100_000)
            self.assertEqual(transport.ready_metadata["device_boot_id"], "b")
            transport.finalize(1)

    def test_paid_exchange_timeout_fails_closed(self) -> None:
        code = (
            "import sys,time; "
            "sys.stderr.write('LAUNCHER_READY {\\\"device_boot_id\\\":\\\"b\\\"}\\n'); "
            "sys.stderr.flush(); sys.stdin.buffer.read(); time.sleep(2)"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            transport = PreparedSubprocessTransport((sys.executable, "-c", code), path, 2)
            reply = transport.exchange(request(50_000), b"{}\n")
            self.assertEqual((reply.state, reply.payload), ("timed_out", b""))


if __name__ == "__main__":
    unittest.main()
