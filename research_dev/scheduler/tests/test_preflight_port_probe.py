"""The preflight's raw endpoint port probe (2026-10-06): TIME_WAIT is not a conflict."""

import socket
import unittest

from research_dev.scheduler.adapters.snapshot import EndpointRuntimeSample
from research_dev.scheduler.campaigns.burstgpt import preflight

NOT_READY = EndpointRuntimeSample(health="unreachable", slots_probe="unreachable", free_slots=0)


def _listener() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    return listener


class PortProbeTests(unittest.TestCase):
    def test_time_wait_remnant_of_the_previous_arm_is_not_a_conflict(self) -> None:
        listener = _listener()
        port = listener.getsockname()[1]
        client = socket.create_connection(("127.0.0.1", port))
        server_side, _ = listener.accept()
        server_side.close()   # the server closes first: its side of the connection sits in TIME_WAIT
        listener.close()
        client.close()
        ok, detail = preflight._local_port_available(f"http://127.0.0.1:{port}", NOT_READY)
        self.assertTrue(ok, detail)
        self.assertEqual(detail, "endpoint port is free for canonical launch")

    def test_a_live_listener_still_conflicts(self) -> None:
        listener = _listener()
        try:
            ok, detail = preflight._local_port_available(
                f"http://127.0.0.1:{listener.getsockname()[1]}", NOT_READY
            )
        finally:
            listener.close()
        self.assertFalse(ok)
        self.assertIn("endpoint port conflict", detail)
