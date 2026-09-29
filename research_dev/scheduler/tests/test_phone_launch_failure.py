"""Exercise the actual shell predicates used before USB enumeration."""
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from research_dev.scheduler.adapters.phone_session_ops.transport import _remote_launch_failure


class PhoneLaunchFailureTests(unittest.TestCase):
    def test_start_running_exit_and_terminal(self):
        def shell(command, timeout_s):
            result = subprocess.run(["sh", "-c", command], capture_output=True, text=True, timeout=timeout_s, check=True)
            self.assertEqual(result.stderr, "")
            return result.stdout

        owner = SimpleNamespace(_adb_root=shell)
        with tempfile.TemporaryDirectory(prefix="phone launch '") as directory:
            root = Path(directory)
            self.assertIsNone(_remote_launch_failure(owner, directory))
            (root / "worker.pid").write_text(str(os.getpid()))
            self.assertIsNone(_remote_launch_failure(owner, directory))
            (root / "worker.pid").write_text("99999999")
            self.assertIn("worker exited", _remote_launch_failure(owner, directory))
            (root / "terminal.status").write_text("1\n")
            self.assertIn("status=1", _remote_launch_failure(owner, directory))


if __name__ == "__main__":
    unittest.main()
