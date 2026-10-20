"""Direct-phone USB close retried while a live server still claims the interface (2026-10-06).

pAlpf1pix2 / pAlpc1pix2: the whole relaunch of a one-session OnePlus 15 started while the hot llama-server's
FFN client still held the FunctionFS interface; the single close attempt failed with LIBUSB_ERROR_BUSY and the
90 s restoration window could not succeed. ``_close_until_restored`` retries the close while it is busy.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler.adapters.phone_session_ops import completion

BUSY = PhysicalAdapterError(
    "direct phone USB close rejected every current resident identity: sha256:x: direct phone USB close failed: "
    "FFNUSB_CLOSE status=error detail=USB interface claim failed: LIBUSB_ERROR_BUSY")


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Phone:
    """A phone that comes back on adb once a close went through (or once ``restored_at`` passes)."""

    def __init__(self, clock, busy_attempts, *, close_error=None, restored_at=None):
        self.clock = clock
        self.busy_attempts = busy_attempts
        self.close_error = close_error
        self.restored_at = restored_at
        self.closed = False
        self.close_calls = 0
        self.verify_timeouts = []
        self.configuration = SimpleNamespace(serial="S", adb_port=5037, minimum_usb_speed_mbps=5000)

    def _close_current_usb(self):
        self.close_calls += 1
        if self.close_calls <= self.busy_attempts:
            raise BUSY
        if self.close_error is not None:
            raise self.close_error
        self.closed = True

    def verify(self, *, serial, adb_port, minimum_speed_mbps, timeout_s):
        self.verify_timeouts.append(timeout_s)
        if self.closed or (self.restored_at is not None and self.clock.now >= self.restored_at):
            return "receipt"
        self.clock.now += timeout_s
        raise PhysicalAdapterError("Android USB restoration failed: ADB state is ")


class CloseUntilRestoredTests(unittest.TestCase):
    def run_case(self, phone, clock, window):
        with patch.object(completion, "verify_android_usb_restored", phone.verify):
            return completion._close_until_restored(phone, window, sleep=clock.sleep, clock=clock.clock)

    def test_busy_close_is_retried_until_the_server_releases_the_interface(self):
        clock = FakeClock(); phone = Phone(clock, busy_attempts=3)
        self.assertEqual(self.run_case(phone, clock, 90), "receipt")
        self.assertEqual(phone.close_calls, 4)
        self.assertEqual(clock.sleeps, [1.0, 1.0, 1.0])
        self.assertEqual(phone.verify_timeouts[:3], [0, 0, 0])       # non-blocking checks while busy

    def test_a_server_exit_that_ends_the_worker_restores_without_a_close(self):
        clock = FakeClock(); phone = Phone(clock, busy_attempts=100, restored_at=5.0)
        self.assertEqual(self.run_case(phone, clock, 90), "receipt")
        self.assertFalse(phone.closed)
        self.assertLessEqual(phone.close_calls, 7)

    def test_busy_for_the_whole_window_fails_with_both_errors(self):
        clock = FakeClock(); phone = Phone(clock, busy_attempts=10_000)
        with self.assertRaisesRegex(PhysicalAdapterError, "LIBUSB_ERROR_BUSY.*restoration failed"):
            self.run_case(phone, clock, 10)
        self.assertLessEqual(clock.now, 11.0)
        self.assertGreaterEqual(phone.close_calls, 9)

    def test_any_other_close_error_gets_one_attempt_and_the_full_wait(self):
        clock = FakeClock(); phone = Phone(clock, busy_attempts=0,
                                           close_error=PhysicalAdapterError("direct phone USB close timed out"))
        with self.assertRaisesRegex(PhysicalAdapterError, "timed out.*restoration failed"):
            self.run_case(phone, clock, 90)
        self.assertEqual((phone.close_calls, phone.verify_timeouts, clock.sleeps), (1, [90], []))

    def test_a_successful_first_close_waits_for_adb_once(self):
        clock = FakeClock(); phone = Phone(clock, busy_attempts=0)
        self.assertEqual(self.run_case(phone, clock, 90), "receipt")
        self.assertEqual((phone.close_calls, clock.sleeps), (1, []))


if __name__ == "__main__":
    unittest.main()
