"""Native gate for S2a: a llama-server that lost one FFN helper keeps serving and re-attaches it.

Elastic phones S2a (reports/20260925-elastic-phones/SPEC.md, S2A_SERVER.diff): the real server with
two local CPU FFN workers (TCP) under runtime control, helper ``op15`` on layers 0-1 and ``pixel``
on layers 2-3.

1. A request decodes with both helpers; the ``pixel`` worker is terminated mid-stream. The decode
   aborts ("Compute aborted", every processing slot errored), the server names the helper at once
   (``S41SERVERFFNERROR helper=pixel``) and KEEPS RUNNING.
2. Masked out: the next policy apply closes the failed session (``S41SERVERFFNRESET cause=failed``,
   no old error text); a host-only request and a request whose policy owns only op15's layers
   complete on the same process. A policy that owns the pixel's layers while its worker is down is
   refused (the reconnect fails, the error is latched again) and the request continues.
3. Re-attached: the pixel worker restarts on the same port; the next policy that owns its layers
   reconnects and calls reach layers 2-3 again. A restarted worker serving other weights (another
   weight hash) is refused.
4. A worker that restarted while this server kept its helper idle (connected, not failed) is found
   closed by the next policy that owns it and reconnected before any decode reaches it.

Binaries come from ``build-cpu/bin`` (override with ``S42_LLAMA_BUILD_BIN``); the test is skipped
when they are absent or the server predates S2a (no ``S41SERVERFFNCAPS`` line).
"""
from __future__ import annotations

import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SCHEDULER_DIR = TESTS_DIR.parent
REPO_ROOT = SCHEDULER_DIR.parents[1]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
if str(REPO_ROOT / "gguf-py") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from tiny_llama_gguf import N_EMBD, N_FF, write_tiny_llama_gguf  # noqa: E402

BIN_DIR = Path(os.environ.get("S42_LLAMA_BUILD_BIN", REPO_ROOT / "build-cpu" / "bin"))
SERVER = BIN_DIR / "llama-server"
WORKER = BIN_DIR / "llama-ffn-split-worker"
SHARD_TOOL = SCHEDULER_DIR / "native" / "ffn_shard_gguf.py"

MAX_TOKENS = 16
PROMPT = [5, 9, 17, 33, 65, 3, 7, 11]
COLUMNS = 64
UNION_MASK, OP15_MASK, PIXEL_MASK = 0b1111, 0b0011, 0b1100
CAPS = "S41SERVERFFNCAPS helper_mask_out=1 helper_reconnect_tcp=1"
CALL = re.compile(r"S41SERVERFFNCALL (?:context=\S+ )?request=(\d+) layer=(\d+) tokens=(\d+) columns=(\d+)")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _request(port: int, method: str, path: str, body: dict | None = None, timeout: float = 60):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method, path, None if body is None else json.dumps(body),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"{}")
    finally:
        connection.close()


class _Stream:
    """One streamed /completion under a scheduler request id; ``chunks`` are its data rows."""

    def __init__(self, port: int, request_id: str, n_predict: int) -> None:
        self.port, self.request_id = port, request_id
        self.connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        self.connection.request("POST", "/completion", json.dumps({
            "prompt": PROMPT, "n_predict": n_predict, "temperature": 0, "ignore_eos": True,
            "stream": True, "cache_prompt": False, "id_slot": 0, "return_tokens": True}),
            {"Content-Type": "application/json", "X-Scheduler-Request-ID": request_id})
        self.response = self.connection.getresponse()
        self.chunks: list[dict] = []
        self.generation = 0

    def read(self, count: int | None = None) -> list[dict]:
        """``count`` more data rows (None: until the stream ends)."""
        rows = []
        while count is None or len(rows) < count:
            line = self.response.readline()
            if not line:
                break
            if line.startswith(b"data: "):
                row = json.loads(line[6:])
                rows.append(row)
                self.chunks.append(row)
                if row.get("stop") or "error" in row:
                    break
        return rows

    def control(self, layer_mask: int, columns: int = COLUMNS) -> dict:
        self.generation += 1
        status, body = _request(self.port, "POST", "/v1/chat/completions/control", {
            "action": "ffn_split", "request_id": self.request_id, "slot_id": 0,
            "plan_generation": self.generation, "policy_hash": "sha256:" + f"{self.generation:064x}",
            "layer_mask": layer_mask if columns else 0, "columns": columns, "enabled": columns != 0})
        if status != 200:
            raise AssertionError(f"control status {status}: {body}")
        return body

    def close(self) -> None:
        self.connection.close()

    @property
    def error(self) -> str | None:
        rows = [row["error"] for row in self.chunks if "error" in row]
        return None if not rows else json.dumps(rows[-1])

    @property
    def stopped(self) -> bool:
        return any(row.get("stop") for row in self.chunks)


class HelperMaskOutNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SERVER.exists() or not WORKER.exists():
            raise unittest.SkipTest(f"native binaries missing under {BIN_DIR}")
        cls.tmp = Path(tempfile.mkdtemp(prefix="s2a-mask-out-"))
        cls.model = write_tiny_llama_gguf(cls.tmp / "tiny.gguf", n_layer=4, seed=5, with_tokenizer=True)
        cls.parent = _sha256(cls.model)
        cls.shards = cls.tmp / "shards"
        subprocess.run(
            [sys.executable, str(SHARD_TOOL), str(cls.model), "--parent-sha256", cls.parent,
             "--out-dir", str(cls.shards), "--shard", f"HTP0=0-1:{N_FF}", "--shard", f"HTP1=2-3:{N_FF}",
             "--verify-parent"],
            check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "gguf-py")},
        )
        # the same geometry with other weights (another seed): its worker's weight hash differs
        cls.other_model = write_tiny_llama_gguf(cls.tmp / "other.gguf", n_layer=4, seed=6, with_tokenizer=True)
        cls.processes = []

    @classmethod
    def tearDownClass(cls) -> None:
        for process, log in getattr(cls, "processes", []):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            log.close()

    @classmethod
    def _worker(cls, name: str, shard: str, layers: str, port: int) -> subprocess.Popen:
        log_path = cls.tmp / f"worker-{name}.log"
        log = log_path.open("w")
        model = cls.shards / shard if not Path(shard).is_absolute() else Path(shard)
        process = subprocess.Popen(
            [str(WORKER), "-m", str(model), "--artifact-sha256", cls.parent, "--layers", layers,
             "--columns", str(N_FF), "--column-quantum", "64", "--backend", "CPU", "--port", str(port),
             "--bind", "127.0.0.1", "--f16-io", "--max-tokens", str(MAX_TOKENS)],
            stdout=log, stderr=subprocess.STDOUT,
        )
        cls.processes.append((process, log))
        deadline = time.monotonic() + 30
        while "ready backend" not in log_path.read_text():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"worker {name} not ready: " + log_path.read_text())
            time.sleep(0.05)
        return process

    def _server(self, ports: dict[str, int]) -> tuple[subprocess.Popen, Path, int]:
        env = {key: value for key, value in os.environ.items() if not key.startswith("S41_SERVER_")}
        env.update({
            "S41_SERVER_FFN_ARTIFACT_SHA256": self.parent, "S41_SERVER_FFN_ACTIVATION": "swiglu",
            "S41_SERVER_FFN_COLUMNS": str(N_FF), "S41_SERVER_FFN_F16_IO": "1",
            "S41_SERVER_FFN_LAYER_MASK": str(UNION_MASK), "S41_SERVER_FFN_N_EMBD": str(N_EMBD),
            "S41_SERVER_FFN_TIMEOUT_MS": "5000", "S41_SERVER_FFN_RUNTIME_CONTROL": "1",
            "S41_SERVER_FFN_MAX_TOKENS": str(MAX_TOKENS), "S41_SERVER_FFN_HELPERS": "2",
        })
        for index, (label, mask) in enumerate((("op15", OP15_MASK), ("pixel", PIXEL_MASK))):
            prefix = f"S41_SERVER_FFN_HELPER{index}_"
            env.update({prefix + "LABEL": label, prefix + "LAYER_MASK": str(mask), prefix + "TRANSPORT": "tcp",
                        prefix + "HOST": "127.0.0.1", prefix + "PORT": str(ports[label])})
        port = _free_port()
        log_path = self.tmp / "server.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [str(SERVER), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(port),
                 "--parallel", "1", "--ctx-size", "1024", "--batch-size", str(MAX_TOKENS),
                 "--ubatch-size", str(MAX_TOKENS), "--n-gpu-layers", "0", "--threads", "2", "--no-warmup"],
                env=env, stdout=log, stderr=subprocess.STDOUT)
        self.addCleanup(lambda: process.poll() is None and (process.terminate(), process.wait(timeout=30)))
        deadline = time.monotonic() + 60
        while True:
            if process.poll() is not None:
                self.fail("server exited at startup: " + log_path.read_text()[-3000:])
            try:
                if _request(port, "GET", "/health", timeout=1)[0] == 200:
                    return process, log_path, port
            except (OSError, http.client.HTTPException, ValueError):
                pass
            if time.monotonic() > deadline:
                process.kill()
                self.fail("server never became ready: " + log_path.read_text()[-3000:])
            time.sleep(0.05)

    def _calls(self, log_path: Path, start: int = 0) -> list[tuple[int, int]]:
        return [(int(request), int(layer))
                for request, layer, *_ in CALL.findall(log_path.read_text()[start:])]

    def test_lost_helper_is_masked_out_then_reattached_on_the_live_server(self) -> None:
        ports = {"op15": _free_port(), "pixel": _free_port()}
        self._worker("op15", "HTP0.ffn.gguf", "0-1", ports["op15"])
        pixel = self._worker("pixel", "HTP1.ffn.gguf", "2-3", ports["pixel"])
        server, log_path, port = self._server(ports)
        if CAPS not in log_path.read_text():
            self.skipTest("llama-server predates S2a helper recovery (apply S2A_SERVER.diff)")

        # 1. both helpers serve, then the pixel worker dies mid-stream
        first = _Stream(port, "s2a-first", 700)
        try:
            first.read(1)
            self.assertTrue(first.control(UNION_MASK)["success"])
            first.read(5)
            self.assertEqual({layer for _, layer in self._calls(log_path)}, {0, 1, 2, 3})
            pixel.send_signal(signal.SIGTERM)
            pixel.wait(timeout=10)
            first.read()
        finally:
            first.close()
        self.assertFalse(first.stopped)
        self.assertIn("Compute aborted", first.error or "")
        self.assertIsNone(server.poll())
        self.assertEqual(_request(port, "GET", "/health")[0], 200)
        text = log_path.read_text()
        self.assertEqual(len(re.findall(r"^S41SERVERFFNERROR helper=pixel detail=", text, re.M)), 1)
        self.assertNotIn("S41SERVERFFNERROR helper=op15", text)

        # 2. masked out: host-only and op15-only requests complete on the same process
        mark = len(log_path.read_text())
        host = _Stream(port, "s2a-host", 40)
        try:
            host.read()
        finally:
            host.close()
        self.assertTrue(host.stopped, host.error)
        self.assertEqual(self._calls(log_path, mark), [])
        resets = re.findall(r"^S41SERVERFFNRESET .*$", log_path.read_text()[mark:], re.M)
        self.assertEqual(resets, ["S41SERVERFFNRESET helper=pixel cause=failed resets=1"])

        mark = len(log_path.read_text())
        masked = _Stream(port, "s2a-masked", 60)
        try:
            masked.read(1)
            refused = masked.control(UNION_MASK)
            self.assertFalse(refused["success"], refused)
            self.assertIn("helper pixel:", refused["message"])
            self.assertTrue(masked.control(OP15_MASK)["success"])
            masked.read()
        finally:
            masked.close()
        self.assertTrue(masked.stopped, masked.error)
        calls = self._calls(log_path, mark)
        self.assertTrue(calls)
        self.assertEqual({layer for _, layer in calls}, {0, 1})
        # the refused reconnect latched its error; the op15-only apply closed it again
        self.assertIn("S41SERVERFFNRESET helper=pixel cause=failed resets=2", log_path.read_text()[mark:])

        # 3a. a worker with other weights on the pixel's port is refused (weight hash differs)
        other = self._worker("pixel-other-weights", str(self.other_model), "2-3", ports["pixel"])
        mark = len(log_path.read_text())
        foreign = _Stream(port, "s2a-foreign", 40)
        try:
            foreign.read(1)
            refused = foreign.control(UNION_MASK)
            self.assertFalse(refused["success"], refused)
            self.assertIn("helper pixel: FFN split worker weights differ from the session it replaces",
                          refused["message"])
            foreign.read()
        finally:
            foreign.close()
        self.assertTrue(foreign.stopped, foreign.error)
        self.assertEqual({layer for _, layer in self._calls(log_path, mark)} & {2, 3}, set())
        other.send_signal(signal.SIGTERM)
        other.wait(timeout=10)
        # 3b. the pixel worker restarts on its port: the next policy that owns it reconnects
        self._worker("pixel-restarted", "HTP1.ffn.gguf", "2-3", ports["pixel"])
        mark = len(log_path.read_text())
        rejoined = _Stream(port, "s2a-rejoined", 60)
        try:
            rejoined.read(1)
            accepted = rejoined.control(UNION_MASK)
            self.assertTrue(accepted["success"], accepted)
            rejoined.read()
        finally:
            rejoined.close()
        self.assertTrue(rejoined.stopped, rejoined.error)
        self.assertEqual({layer for _, layer in self._calls(log_path, mark)}, {0, 1, 2, 3})
        self.assertIsNone(server.poll())

        server.terminate()
        server.wait(timeout=30)
        text = log_path.read_text()
        summaries = {json.loads(line.split(" ", 1)[1])["helper"]: json.loads(line.split(" ", 1)[1])["status"]
                     for line in text.splitlines() if line.startswith("S41SERVERFFN {")}
        # the re-attached session ended healthy: no shutdown error line for it, a sticky reset summary
        self.assertEqual(summaries, {"op15": "ok", "pixel": "ok"})
        self.assertEqual(len(re.findall(r"^S41SERVERFFNERROR ", text, re.M)), 1)
        (summary,) = re.findall(r"^S41SERVERFFNRESET helper=pixel summary .*$", text, re.M)
        self.assertEqual(summary, "S41SERVERFFNRESET helper=pixel summary resets=3 "
                                  "last_error=FFN split worker weights differ from the session it replaces")

    def test_an_idle_helper_whose_worker_restarted_reconnects_when_owned_again(self) -> None:
        """The pixel worker restarts while this server does not use it (its client is idle, not
        failed): the next policy that owns the pixel's layers finds the closed session, resets it
        and reconnects instead of failing the next decode."""
        ports = {"op15": _free_port(), "pixel": _free_port()}
        self._worker("op15-idle", "HTP0.ffn.gguf", "0-1", ports["op15"])
        pixel = self._worker("pixel-idle", "HTP1.ffn.gguf", "2-3", ports["pixel"])
        server, log_path, port = self._server(ports)
        if CAPS not in log_path.read_text():
            self.skipTest("llama-server predates S2a helper recovery (apply S2A_SERVER.diff)")
        first = _Stream(port, "s2a-idle-first", 40)
        try:
            first.read(1)
            self.assertTrue(first.control(UNION_MASK)["success"])
            first.read(3)
            self.assertTrue(first.control(OP15_MASK)["success"])  # the pixel is idle from here
            first.read()
        finally:
            first.close()
        self.assertTrue(first.stopped, first.error)
        pixel.send_signal(signal.SIGTERM)
        pixel.wait(timeout=10)
        self._worker("pixel-idle-restarted", "HTP1.ffn.gguf", "2-3", ports["pixel"])
        mark = len(log_path.read_text())
        second = _Stream(port, "s2a-idle-second", 40)
        try:
            second.read(1)
            accepted = second.control(UNION_MASK)
            self.assertTrue(accepted["success"], accepted)
            second.read()
        finally:
            second.close()
        self.assertTrue(second.stopped, second.error)
        text = log_path.read_text()[mark:]
        self.assertIn("S41SERVERFFNRESET helper=pixel cause=peer_closed resets=1", text)
        self.assertEqual({layer for _, layer in self._calls(log_path, mark)}, {0, 1, 2, 3})
        self.assertNotIn("S41SERVERFFNERROR", log_path.read_text())
        self.assertIsNone(server.poll())


if __name__ == "__main__":
    unittest.main()
