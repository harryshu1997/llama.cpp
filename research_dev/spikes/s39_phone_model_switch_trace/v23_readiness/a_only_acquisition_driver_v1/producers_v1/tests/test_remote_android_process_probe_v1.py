#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "remote_android_process_probe_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "remote_android_process_probe_v1",
    SOURCE,
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def encoded(value: bytes) -> str:
    return value.hex()


class RemoteAndroidProcessProbeTests(unittest.TestCase):
    def setUp(self):
        self.boot_id = "11111111-1111-4111-8111-111111111111"
        self.adb_port = 5038
        self.adb_selector = "172.20.173.218:5555"
        self.executable = "/data/local/tmp/llama-stage-direct-relay"
        self.argv = [
            self.executable,
            "--listen",
            "12345",
            "--head",
            "127.0.0.1:1001",
            "--tail",
            "192.0.2.2:1002",
            "--emit-direct-frames",
        ]

    def remote_output(self, pid=123, start_ticks=456):
        cmdline = b"\x00".join(
            value.encode("ascii") for value in self.argv
        ) + b"\x00"
        stat_fields = ["S", *["0"] * 18, str(start_ticks), *["0"] * 4]
        stat_raw = (
            f"{pid} (relay worker) {' '.join(stat_fields)}\n"
        ).encode("ascii")
        return (
            f"B {encoded((self.boot_id + chr(10)).encode('ascii'))}\n"
            f"P {pid} {encoded(self.executable.encode('ascii'))} "
            f"{encoded(cmdline)} {encoded(stat_raw)}\n"
        ).encode("ascii")

    def test_exact_remote_process_is_selected(self):
        value = probe.parse_remote(
            self.remote_output(),
            self.boot_id,
            self.executable,
            self.argv,
            12345,
            self.adb_port,
            self.adb_selector,
        )
        self.assertEqual(value["pid"], 123)
        self.assertEqual(value["start_ticks"], 456)
        self.assertEqual(value["argv"], self.argv)
        self.assertEqual(value["adb_port"], self.adb_port)
        self.assertEqual(value["adb_selector"], self.adb_selector)

    def test_wrong_argv_or_executable_is_rejected(self):
        for executable, argv in (
            ("/wrong", self.argv),
            (self.executable, [*self.argv, "--extra"]),
        ):
            with self.subTest(executable=executable, argv=argv):
                with self.assertRaisesRegex(
                    probe.ProbeError,
                    "E_REMOTE_PROCESS_COUNT",
                ):
                    probe.parse_remote(
                        self.remote_output(),
                        self.boot_id,
                        executable,
                        argv,
                        12345,
                        self.adb_port,
                        self.adb_selector,
                    )

    def test_duplicate_exact_process_is_rejected(self):
        raw = self.remote_output()
        duplicate = raw + raw.splitlines(keepends=True)[1]
        with self.assertRaisesRegex(
            probe.ProbeError,
            "E_REMOTE_PROCESS_COUNT",
        ):
            probe.parse_remote(
                duplicate,
                self.boot_id,
                self.executable,
                self.argv,
                12345,
                self.adb_port,
                self.adb_selector,
            )

    def test_adb_argv_binds_server_port_and_wifi_selector(self):
        argv = probe.adb_argv(
            Path("/usr/bin/adb"),
            self.adb_port,
            self.adb_selector,
        )
        self.assertEqual(
            argv[:5],
            [
                "/usr/bin/adb",
                "-P",
                "5038",
                "-s",
                "172.20.173.218:5555",
            ],
        )
        for port, selector, message in (
            (0, self.adb_selector, "E_ADB_PORT"),
            (65536, self.adb_selector, "E_ADB_PORT"),
            (self.adb_port, "", "E_ADB_SELECTOR"),
            (self.adb_port, "bad\nselector", "E_ADB_SELECTOR"),
        ):
            with self.subTest(port=port, selector=selector):
                with self.assertRaisesRegex(probe.ProbeError, message):
                    probe.adb_argv(Path("/usr/bin/adb"), port, selector)


if __name__ == "__main__":
    unittest.main()
