"""The session wait loop must tolerate the adb launcher exiting non-zero.

The phone drops its ADB link while switching USB function, so on some hosts
the ``adb shell`` launcher exits non-zero before the FunctionFS device
enumerates.  That exit is not evidence of a failed launch.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler.adapters.phone_session_ops import launch


class ExitedLauncher:
    def __init__(self, returncode=1, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.communicate_calls = 0

    def poll(self):
        return self.returncode

    def communicate(self):
        self.communicate_calls += 1
        return "", self.stderr


def controller(*, launch_timeout_s, remote_failure=None):
    return SimpleNamespace(
        configuration=SimpleNamespace(
            launch_timeout_s=launch_timeout_s, minimum_usb_speed_mbps=5000
        ),
        _remote_start_failure_detail=lambda remote_root, launch_log: remote_failure,
        _validate_live_transport_identity=lambda *args: None,
    )


TRANSPORT = SimpleNamespace(vendor_id=0x18D1, product_id=0x2D00)


class PhoneLaunchWaitTests(unittest.TestCase):
    def wait(self, owner, process, probe):
        with patch.object(launch, "probe_functionfs_usb_device", probe), patch.object(
            launch.time, "sleep", lambda seconds: None
        ):
            return launch._wait_for_phone_session(
                owner, process, TRANSPORT, (object(),), "kernel", "usb", "/root", "/root/launch.log",
                lambda: None,
            )

    def test_launcher_exit_before_enumeration_is_tolerated(self):
        misses = []

        def probe(*, vendor_id, product_id):
            if len(misses) < 3:
                misses.append((vendor_id, product_id))
                raise PhysicalAdapterError("FunctionFS USB did not enumerate")
            return SimpleNamespace(negotiated_speed_mbps=10000)

        process = ExitedLauncher(returncode=255)
        usb, last_error = self.wait(controller(launch_timeout_s=5), process, probe)
        self.assertIsNotNone(usb)
        self.assertEqual(len(misses), 3)
        self.assertEqual(misses[0], ("18d1", "2d00"))
        self.assertEqual(process.communicate_calls, 1)

    def test_launcher_exit_without_enumeration_reports_both(self):
        def probe(*, vendor_id, product_id):
            raise PhysicalAdapterError("FunctionFS USB did not enumerate")

        process = ExitedLauncher(returncode=1, stderr="adb: device offline")
        usb, last_error = self.wait(controller(launch_timeout_s=0.3), process, probe)
        self.assertIsNone(usb)
        self.assertIn("phone session launcher exited: adb: device offline", last_error)
        self.assertIn("FunctionFS USB did not enumerate", last_error)

    def test_remote_failure_still_ends_the_wait(self):
        def probe(*, vendor_id, product_id):
            raise PhysicalAdapterError("FunctionFS USB did not enumerate")

        owner = controller(launch_timeout_s=30, remote_failure="worker exited: status=1")
        with patch.object(launch.time, "monotonic", side_effect=[0.0, 0.0, 1.5, 1.5, 1.6, 1.6, 1.7]):
            usb, last_error = self.wait(owner, ExitedLauncher(), probe)
        self.assertIsNone(usb)
        self.assertEqual(last_error, "worker exited: status=1")


if __name__ == "__main__":
    unittest.main()
