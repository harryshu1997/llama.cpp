"""restore_android_usb.sh: the adbd watchdog waits for the host's acknowledgement.

Fault run p6h2flt (2026-10-04): after the gadget switch back to Android the ADB interface was enumerated but
adbd never answered, so "Android USB restoration failed: ADB state is" ended the run and the rig idled for
ten hours until a replug (which did not help either). The script now waits for ``host-ack`` (written by the
host over adb once ``adb get-state`` works), restarts adbd without it, then cycles the USB configuration.
Run here with stubbed setprop/getprop and a fake configfs tree.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "adapters" / "native" / "restore_android_usb.sh"


class RestoreWatchdogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.session = root / "session"
        self.g1 = root / "g1"
        self.g2 = root / "g2"
        for g in (self.g1, self.g2):
            (g / "configs" / "b.1").mkdir(parents=True)
            (g / "functions").mkdir()
        (self.g2 / "UDC").write_text("udc0\n")
        (self.g1 / "UDC").write_text("")
        self.session.mkdir()
        (self.session / "android-gadget.path").write_text(str(self.g1))
        (self.session / "functionfs-gadget.path").write_text(str(self.g2))
        (self.session / "phone-udc.name").write_text("udc0")
        (self.session / "functionfs-root.path").write_text(str(root / "ffs"))
        (self.session / "android-usb-config").write_text("mtp,adb")
        (self.session / "host-ack").write_text("stale")
        self.calls = root / "setprop.log"
        stubs = root / "bin"
        stubs.mkdir()
        (stubs / "setprop").write_text(
            "#!/bin/sh\necho \"$1 $2\" >> \"$SETPROP_LOG\"\n"
            "if [ \"$1\" = sys.usb.config ] && [ \"$2\" != none ]; then echo udc0 > \"$G1_UDC\"; fi\n"
            "if [ \"$1\" = sys.usb.config ] && [ \"$2\" = none ]; then : > \"$G1_UDC\"; fi\n")
        (stubs / "getprop").write_text("#!/bin/sh\ncase \"$1\" in init.svc.adbd) echo running;; sys.usb.state) echo mtp,adb;; esac\n")
        for f in ("setprop", "getprop"):
            (stubs / f).chmod((stubs / f).stat().st_mode | stat.S_IEXEC)
        self.env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "SETPROP_LOG": str(self.calls),
                    "G1_UDC": str(self.g1 / "UDC"), "S41_RESTORE_ACK_TIMEOUT_S": "2"}

    def run_script(self):
        completed = subprocess.run(["sh", str(SCRIPT), str(self.session)], env=self.env, capture_output=True,
                                   text=True, timeout=30)
        return completed.returncode

    def wait_log(self, needle, timeout_s=15.0):
        log = self.session / "restore-watchdog.log"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if log.exists() and needle in log.read_text():
                return log.read_text()
            time.sleep(0.1)
        self.fail(f"watchdog log lacks {needle!r}: " + (log.read_text() if log.exists() else "<absent>"))

    def test_the_host_acknowledgement_ends_the_watch_without_touching_adbd(self):
        def ack():
            time.sleep(0.4)
            (self.session / "host-ack").write_text("ok")
        threading.Thread(target=ack, daemon=True).start()
        self.assertEqual(self.run_script(), 0)
        self.assertFalse((self.session / "host-ack").exists() and (self.session / "host-ack").read_text() == "stale")
        log = self.wait_log("host-ack after")
        self.assertNotIn("restarting adbd", log)
        self.assertEqual(self.calls.read_text().splitlines(), ["sys.usb.config mtp,adb"])
        self.assertEqual((self.g1 / "UDC").read_text().strip(), "udc0")
        self.assertFalse((self.session / "active").exists())

    def test_without_acknowledgement_adbd_is_restarted_then_the_usb_configuration_is_cycled(self):
        self.assertEqual(self.run_script(), 0)
        log = self.wait_log("gave up")
        self.assertIn("restarting adbd", log)
        self.assertIn("cycling the USB configuration", log)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(calls, ["sys.usb.config mtp,adb", "ctl.restart adbd", "sys.usb.config none",
                                 "sys.usb.config mtp,adb", "ctl.restart adbd"])
        self.assertEqual((self.g1 / "UDC").read_text().strip(), "udc0")

    def test_a_new_session_ends_the_watch(self):
        def start_session():
            time.sleep(0.4)
            (self.session / "active").write_text("")
        threading.Thread(target=start_session, daemon=True).start()
        self.assertEqual(self.run_script(), 0)
        log = self.wait_log("new session active")
        self.assertNotIn("restarting adbd", log)


if __name__ == "__main__":
    unittest.main()
