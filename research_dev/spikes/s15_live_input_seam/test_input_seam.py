#!/usr/bin/env python3

from __future__ import annotations

import os
import json
import selectors
import subprocess
import time
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BINARY = Path(os.environ.get(
    "LAYERSPLIT_TEST_BINARY", ROOT / "build-cpu/bin/llama-layersplit",
))
MODEL = Path(os.environ.get(
    "LAYERSPLIT_TEST_MODEL", ROOT / "build-phone-pim/tinyllamas/stories15M-q4_0.gguf",
))


class InputSeamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(BINARY.is_file())
        self.assertTrue(MODEL.is_file())
        self.env = dict(os.environ)
        self.env["LD_LIBRARY_PATH"] = str(BINARY.parent)
        self.env["LLAMA_LAYER_START"] = "1"

    def command(self) -> list[str]:
        return [
            str(BINARY), "-m", str(MODEL), "-ngl", "0",
            "--mode", "pipedriver", "--host", "127.0.0.1", "--port", "6553",
            "--prompt-after-load", "-n", "1", "--driver-requests", "1",
            "--driver-batch", "1", "--driver-context", "64",
            "--driver-max-prefill", "8",
        ]

    def start_until_ready(self) -> tuple[subprocess.Popen, list[str]]:
        process = subprocess.Popen(
            self.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=self.env, bufsize=0,
        )
        selector = selectors.DefaultSelector()
        selector.register(process.stderr, selectors.EVENT_READ)
        captured = b""
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            for key, _ in selector.select(timeout=0.2):
                chunk = os.read(key.fileobj.fileno(), 4096)
                captured += chunk
                if b"DRIVER_INPUT_READY " in captured:
                    selector.close()
                    return process, captured.decode("utf-8", errors="replace").splitlines()
        selector.close()
        process.kill()
        process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
        lines = captured.decode("utf-8", errors="replace").splitlines()
        self.fail(f"missing input-ready marker: {lines[-10:]}")

    def test_prompt_is_consumed_only_after_ready(self) -> None:
        process, lines = self.start_until_ready()
        self.assertIsNone(process.poll())
        self.assertEqual(
            lines[-1],
            'DRIVER_INPUT_READY {"schema":"layersplit-driver-input-v1","max_prompt_bytes":16384}',
        )
        process.stdin.write(b"Explain batching.\n")
        process.stdin.flush()
        process.stdin.close()
        process.wait(timeout=20)
        remainder = process.stderr.read().decode("utf-8", errors="replace").splitlines()
        self.assertIn(
            'DRIVER_INPUT_ACCEPTED {"schema":"layersplit-driver-input-v1","prompt_bytes":17}',
            remainder,
        )
        self.assertEqual(process.returncode, 3)  # no stage server in this seam-only test
        process.stdout.close()
        process.stderr.close()

    def test_empty_prompt_fails_before_connect(self) -> None:
        process, _ = self.start_until_ready()
        stdout, stderr = process.communicate(b"\n", timeout=20)
        rendered = stderr.decode("utf-8", errors="replace")
        self.assertEqual(stdout, b"")
        self.assertEqual(process.returncode, 1)
        self.assertIn("prompt-after-load received an empty prompt", rendered)
        self.assertNotIn("connect A", rendered)

    def test_command_line_prompt_cannot_bypass_seam(self) -> None:
        command = self.command() + ["-p", "already known"]
        process = subprocess.run(
            command, capture_output=True, text=True, env=self.env, timeout=10,
        )
        self.assertEqual(process.returncode, 1)
        self.assertIn("forbids -p", process.stderr)
        self.assertNotIn("DRIVER_INPUT_READY", process.stderr)

    def test_non_pipedriver_mode_rejected(self) -> None:
        command = [
            str(BINARY), "-m", "/nonexistent", "--mode", "monodriver",
            "--prompt-after-load",
        ]
        process = subprocess.run(
            command, capture_output=True, text=True, env=self.env, timeout=10,
        )
        self.assertEqual(process.returncode, 1)
        self.assertIn("requires pipedriver", process.stderr)
        self.assertNotIn("unable to load model", process.stderr)

    def test_detach_requires_a_supported_driver(self) -> None:
        command = [
            str(BINARY), "-m", "/nonexistent", "--mode", "pipedriver",
            "--host", "127.0.0.1", "--port", "6553", "-p", "test",
            "--driver-batch", "1", "--session-end", "detach",
        ]
        process = subprocess.run(
            command, capture_output=True, text=True, env=self.env, timeout=10,
        )
        self.assertEqual(process.returncode, 1)
        self.assertIn("requires batched or parallel-head pipedriver", process.stderr)
        self.assertNotIn("unable to load model", process.stderr)

    def test_batched_driver_accepts_detach_option(self) -> None:
        command = [
            str(BINARY), "-m", str(MODEL), "-ngl", "0",
            "--mode", "pipedriver", "--host", "127.0.0.1", "--port", "6553",
            "-p", "test", "-n", "1", "--driver-requests", "2",
            "--driver-batch", "2", "--driver-context", "64",
            "--driver-max-prefill", "8", "--session-end", "detach",
        ]
        process = subprocess.run(
            command, capture_output=True, text=True, env=self.env, timeout=20,
        )
        self.assertEqual(process.returncode, 3)
        self.assertIn("connect A", process.stderr)
        self.assertNotIn("unknown / incomplete arg", process.stderr)

    def persistent_command(self) -> list[str]:
        return [
            str(BINARY), "-m", str(MODEL), "-ngl", "0",
            "--mode", "pipedriver", "--host", "127.0.0.1", "--port", "6553",
            "-n", "1", "--driver-batch", "2", "--driver-context", "64",
            "--driver-max-prefill", "8", "--persistent-jsonl",
        ]

    def test_persistent_driver_requires_placement_observation(self) -> None:
        env = dict(self.env)
        env.pop("LAYERSPLIT_PLACEMENT_CERT", None)
        process = subprocess.run(
            self.persistent_command(), input=b"", capture_output=True,
            env=env, timeout=10,
        )
        self.assertEqual(process.returncode, 1)
        self.assertIn(b"requires LAYERSPLIT_PLACEMENT_CERT=1", process.stderr)
        self.assertNotIn(b"PERSISTENT_DRIVER_READY", process.stderr)

    def test_persistent_driver_rejects_duplicate_command_key(self) -> None:
        env = dict(self.env)
        env["LAYERSPLIT_PLACEMENT_CERT"] = "1"
        payload = (
            b'{"schema":"layersplit-persistent-command-v1","launch_id":1,'
            b'"launch_id":2,"prompt":"test","n_gen":1,"request_count":2,'
            b'"session_end":"STOP"}\n'
        )
        process = subprocess.run(
            self.persistent_command(), input=payload, capture_output=True,
            env=env, timeout=30,
        )
        self.assertEqual(process.returncode, 3)
        self.assertEqual(process.stdout, b"")
        self.assertIn(b"PERSISTENT_DRIVER_READY", process.stderr)
        self.assertIn(b"duplicate persistent command key", process.stderr)

    def test_persistent_driver_rejects_unsafe_shape_before_loading(self) -> None:
        env = dict(self.env)
        env["LAYERSPLIT_PLACEMENT_CERT"] = "1"
        command = self.persistent_command()
        command[command.index(str(MODEL))] = "/nonexistent"
        command.extend(["--port2", "6554"])
        process = subprocess.run(
            command, input=b"", capture_output=True, env=env, timeout=10,
        )
        self.assertEqual(process.returncode, 1)
        self.assertIn(b"requires a batched mono or single-stage pipedriver", process.stderr)
        self.assertNotIn(b"unable to load model", process.stderr)

    def test_persistent_monodriver_reuses_pid_and_resets_kv(self) -> None:
        env = dict(self.env)
        env.pop("LLAMA_LAYER_START", None)
        env.pop("LLAMA_LAYER_END", None)
        env["LAYERSPLIT_PLACEMENT_CERT"] = "1"
        command = [
            str(BINARY), "-m", str(MODEL), "-ngl", "0",
            "--mode", "monodriver", "-n", "2", "--driver-batch", "2",
            "--driver-context", "64", "--driver-max-prefill", "8",
            "--persistent-jsonl",
        ]
        payload = b"".join((
            b'{"schema":"layersplit-persistent-command-v1","launch_id":1,'
            b'"prompt":"Hi","n_gen":2,"request_count":2,"session_end":"DETACH"}\n',
            b'{"schema":"layersplit-persistent-command-v1","launch_id":2,'
            b'"prompt":"Hi","n_gen":2,"request_count":2,"session_end":"STOP"}\n',
        ))
        process = subprocess.run(
            command, input=payload, capture_output=True, env=env, timeout=60,
        )
        self.assertEqual(process.returncode, 0, process.stderr.decode(errors="replace"))
        replies = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual([reply["launch_id"] for reply in replies], [1, 2])
        self.assertEqual(replies[0]["host_pid"], replies[1]["host_pid"])
        self.assertEqual(replies[0]["token_ids"], replies[1]["token_ids"])
        self.assertEqual(replies[0]["session_end"], "DETACH")
        self.assertEqual(replies[1]["session_end"], "STOP")
        placements = [
            json.loads(line.removeprefix("PLACEMENTCERT "))
            for line in process.stderr.decode().splitlines()
            if line.startswith("PLACEMENTCERT ")
        ]
        self.assertEqual(len(placements), 2)
        self.assertTrue(all(item["role"] == "monodriver" for item in placements))
        self.assertTrue(all(item["status"] == "SCHEDULED_PLACEMENT_OK" for item in placements))


if __name__ == "__main__":
    unittest.main()
