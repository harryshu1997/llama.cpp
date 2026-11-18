"""The direct-USB bridge session as an adapter: re-export identity and its two opt-in additions."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler.adapters import phone_functionfs_bridge_session as adapter  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import weight_stream_microbench as wsm  # noqa: E402

import test_ws16_weight_stream_microbench as ws16  # noqa: E402

MULTI = {"resident_workers": "/data/local/tmp/bin/resident-workers",
         "resident_router": "/data/local/tmp/bin/resident-router",
         "rows": "S0,S0,sha256:aa,/data/s0.gguf,0-1,32,27511,sha256:bb,10,sha256:cc,sha256:dd,1;"
                 "S1,S1,sha256:aa,/data/s1.gguf,2-3,32,27512,sha256:bb,10,sha256:cc,sha256:dd,1"}


def make(directory, **options):
    return ws16.FunctionFsSessionTests().make(directory, **options)


def drop_launch_reply(session, returncode):
    """adb loses the phone while the detached launch command returns."""
    run = session._run

    def dropping(command, **keywords):
        if "nohup env" in command[-1]:
            run(command, **keywords)          # the phone did start the session
            raise subprocess.CalledProcessError(returncode, command, output="", stderr="error: closed")
        return run(command, **keywords)

    session._run = dropping


class FunctionFsBridgeAdapterTests(unittest.TestCase):
    def test_tool_reexports_the_adapter_objects(self):
        self.assertIs(wsm.FunctionFsBridgePhoneWorkerSession, adapter.FunctionFsBridgePhoneWorkerSession)
        self.assertIs(wsm.FUNCTIONFS_KEYS, adapter.FUNCTIONFS_KEYS)
        self.assertEqual((adapter.F16_IO_FLAG, adapter.SWIGLU_FLAG), (wsm.F16_IO_FLAG, wsm.SWIGLU_FLAG))

    def test_dropped_launch_reply_fails_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            session, state = make(directory, gadget=[False])
            session._boot_id = "boot-1"
            drop_launch_reply(session, 255)
            with self.assertRaises(subprocess.CalledProcessError):
                session.start(Path(directory) / "worker.log")
            self.assertFalse(session.active)
            self.assertIsNone(session.launch_dropped)

    def test_dropped_launch_reply_is_tolerated_on_request(self):
        with tempfile.TemporaryDirectory() as directory:
            session, state = make(directory, gadget=[False, False, True, True, False])
            session.tolerate_launch_disconnect = True
            session._boot_id = "boot-1"
            drop_launch_reply(session, 255)
            launch = session.start(Path(directory) / "worker.log").to_json()
            self.assertTrue(session.active)
            self.assertEqual(session.launch_dropped, {"stderr": "error: closed", "stdout": ""})
            self.assertEqual(launch["phase"], "launch")
            session.stop(served_calls=10)
            self.assertFalse(session.active)

    def test_other_launch_failures_are_never_tolerated(self):
        with tempfile.TemporaryDirectory() as directory:
            session, state = make(directory, gadget=[False])
            session.tolerate_launch_disconnect = True
            session._boot_id = "boot-1"
            drop_launch_reply(session, 1)
            with self.assertRaises(subprocess.CalledProcessError):
                session.start(Path(directory) / "worker.log")
            self.assertFalse(session.active)

    def test_constructor_keyword_sets_the_tolerance(self):
        session = adapter.FunctionFsBridgePhoneWorkerSession(ws16.functionfs_worker(), ws16.FUNCTIONFS,
                                                             tolerate_launch_disconnect=True)
        self.assertTrue(session.tolerate_launch_disconnect)
        self.assertFalse(adapter.FunctionFsBridgePhoneWorkerSession(
            ws16.functionfs_worker(), ws16.FUNCTIONFS).tolerate_launch_disconnect)

    def test_multi_session_adds_three_arguments_after_the_single_session_ones(self):
        worker = ws16.functionfs_worker()
        root = "/data/local/tmp/ws16-ffs/token"
        single = adapter.FunctionFsBridgePhoneWorkerSession(worker, ws16.FUNCTIONFS)._remote_command(root)
        session = adapter.FunctionFsBridgePhoneWorkerSession(worker, ws16.FUNCTIONFS, multi_session=MULTI)
        multi = session._remote_command(root)
        # a key of that name inside the functionfs section is not read (older harness subclasses own it)
        ignored = adapter.FunctionFsBridgePhoneWorkerSession(worker, dict(ws16.FUNCTIONFS, multi_session=MULTI))
        self.assertEqual(ignored._remote_command(root), single)
        with self.assertRaisesRegex(ValueError, "multi_session needs rows"):
            adapter.FunctionFsBridgePhoneWorkerSession(worker, ws16.FUNCTIONFS,
                                                       multi_session={**MULTI, "rows": ""})
        sha = session.worker.artifact_sha256
        head, tail = single.split(" " + sha + " > ")
        self.assertTrue(multi.startswith(head + " " + sha + " "))
        self.assertTrue(multi.endswith(" > " + tail))
        extra = multi[len(head) + len(sha) + 2:-len(" > " + tail)]
        # the rows hold ';' and must reach the script as one quoted word
        self.assertEqual(extra, MULTI["resident_workers"] + " " + MULTI["resident_router"]
                         + " '" + MULTI["rows"] + "'")

    def test_release_is_the_public_failure_path(self):
        with tempfile.TemporaryDirectory() as directory:
            session, state = make(directory, gadget=[True])
            session._boot_id = "boot-1"
            session.start(Path(directory) / "worker.log")
            self.assertTrue(session.active)
            self.assertEqual(session.release(), [])
            self.assertFalse(session.active)
            self.assertEqual(state["closed"], 1)


if __name__ == "__main__":
    unittest.main()
