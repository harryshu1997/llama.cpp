"""Native gate: a new prompt is not starved by a decoding slot under another FFN policy.

2026-10-08, Qwen3-32B OnePlus arm (reports/20261008-32b-two-phone): the server batches slots only when their FFN
policies are equal, and the rotating policy anchor considered decoding slots only. FFN control reaches a slot in
decode, so a new request's prompt runs under the default (host) policy; while a phone-assisted request decoded, the
anchor was always that request and the new prompt waited inside the server until it finished (server queue 36-94 s
per request, one request at a time over three runs). A slot with a pending prompt now takes its turn as the anchor.

Real server, one local CPU FFN worker (TCP) on all four layers of a tiny model, two slots:
request A decodes under a phone policy (runtime control); request B arrives with a short output and must finish while
A is still decoding. B's tokens equal a solo run of B.

Binaries come from ``build-cpu/bin`` (override with ``S42_LLAMA_BUILD_BIN``); skipped when they are absent.
"""
from __future__ import annotations

import http.client
import json
import os
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

from test_s2a_helper_mask_out_native import _free_port, _request, _sha256  # noqa: E402
from tiny_llama_gguf import N_EMBD, N_FF, write_tiny_llama_gguf  # noqa: E402

BIN_DIR = Path(os.environ.get("S42_LLAMA_BUILD_BIN", REPO_ROOT / "build-cpu" / "bin"))
SERVER = BIN_DIR / "llama-server"
WORKER = BIN_DIR / "llama-ffn-split-worker"
SHARD_TOOL = SCHEDULER_DIR / "native" / "ffn_shard_gguf.py"

MAX_TOKENS = 16
PROMPT = [5, 9, 17, 33, 65, 3, 7, 11]
MASK = 0b1111
COLUMNS = 64
LONG, SHORT = 240, 8  # the tiny model's context is 256 per slot


class _Stream:
    """One streamed /completion on a fixed slot under a scheduler request id."""

    def __init__(self, port: int, request_id: str, slot: int, n_predict: int) -> None:
        self.port, self.request_id, self.slot = port, request_id, slot
        self.connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        self.connection.request("POST", "/completion", json.dumps({
            "prompt": PROMPT, "n_predict": n_predict, "temperature": 0, "ignore_eos": True,
            "stream": True, "cache_prompt": False, "id_slot": slot, "return_tokens": True}),
            {"Content-Type": "application/json", "X-Scheduler-Request-ID": request_id})
        self.response = self.connection.getresponse()
        self.chunks: list[dict] = []

    def read(self, count: int | None = None) -> None:
        rows = 0
        while count is None or rows < count:
            line = self.response.readline()
            if not line:
                return
            if line.startswith(b"data: "):
                row = json.loads(line[6:])
                self.chunks.append(row)
                rows += 1
                if row.get("stop") or "error" in row:
                    return

    def control(self, layer_mask: int) -> dict:
        status, body = _request(self.port, "POST", "/v1/chat/completions/control", {
            "action": "ffn_split", "request_id": self.request_id, "slot_id": self.slot, "plan_generation": 1,
            "policy_hash": "sha256:" + "1" * 64, "layer_mask": layer_mask, "columns": COLUMNS, "enabled": True})
        if status != 200:
            raise AssertionError(f"control status {status}: {body}")
        return body

    @property
    def tokens(self) -> list[int]:
        return [token for row in self.chunks for token in row.get("tokens", [])]

    @property
    def stopped(self) -> bool:
        return any(row.get("stop") for row in self.chunks)


class PromptAnchorNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SERVER.exists() or not WORKER.exists():
            raise unittest.SkipTest(f"native binaries missing under {BIN_DIR}")
        cls.tmp = Path(tempfile.mkdtemp(prefix="prompt-anchor-"))
        cls.model = write_tiny_llama_gguf(cls.tmp / "tiny.gguf", n_layer=4, seed=5, with_tokenizer=True)
        cls.parent = _sha256(cls.model)
        subprocess.run(
            [sys.executable, str(SHARD_TOOL), str(cls.model), "--parent-sha256", cls.parent,
             "--out-dir", str(cls.tmp / "shards"), "--shard", f"HTP0=0-3:{N_FF}", "--verify-parent"],
            check=True, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "gguf-py")})
        cls.processes: list[tuple[subprocess.Popen, object]] = []
        cls.worker_port = _free_port()
        log_path = cls.tmp / "worker.log"
        log = log_path.open("w")
        worker = subprocess.Popen(
            [str(WORKER), "-m", str(cls.tmp / "shards" / "HTP0.ffn.gguf"), "--artifact-sha256", cls.parent,
             "--layers", "0-3", "--columns", str(N_FF), "--column-quantum", "64", "--backend", "CPU",
             "--port", str(cls.worker_port), "--bind", "127.0.0.1", "--f16-io", "--max-tokens", str(MAX_TOKENS)],
            stdout=log, stderr=subprocess.STDOUT)
        cls.processes.append((worker, log))
        deadline = time.monotonic() + 30
        while "ready backend" not in log_path.read_text():
            if worker.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("worker not ready: " + log_path.read_text())
            time.sleep(0.05)

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

    def _server(self) -> tuple[int, Path]:
        env = {key: value for key, value in os.environ.items() if not key.startswith("S41_SERVER_")}
        env.update({
            "S41_SERVER_FFN_ARTIFACT_SHA256": self.parent, "S41_SERVER_FFN_ACTIVATION": "swiglu",
            "S41_SERVER_FFN_COLUMNS": str(N_FF), "S41_SERVER_FFN_F16_IO": "1",
            "S41_SERVER_FFN_LAYER_MASK": str(MASK), "S41_SERVER_FFN_N_EMBD": str(N_EMBD),
            "S41_SERVER_FFN_TIMEOUT_MS": "5000", "S41_SERVER_FFN_RUNTIME_CONTROL": "1",
            "S41_SERVER_FFN_MAX_TOKENS": str(MAX_TOKENS), "S41_SERVER_FFN_HELPERS": "1",
            "S41_SERVER_FFN_HELPER0_LABEL": "op15", "S41_SERVER_FFN_HELPER0_LAYER_MASK": str(MASK),
            "S41_SERVER_FFN_HELPER0_TRANSPORT": "tcp", "S41_SERVER_FFN_HELPER0_HOST": "127.0.0.1",
            "S41_SERVER_FFN_HELPER0_PORT": str(self.worker_port),
        })
        port = _free_port()
        log_path = self.tmp / f"server-{port}.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [str(SERVER), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(port),
                 "--parallel", "2", "--ctx-size", "512", "--batch-size", str(MAX_TOKENS),
                 "--ubatch-size", str(MAX_TOKENS), "--n-gpu-layers", "0", "--threads", "2", "--no-warmup"],
                env=env, stdout=log, stderr=subprocess.STDOUT)
        self.addCleanup(lambda: process.poll() is None and (process.terminate(), process.wait(timeout=30)))
        deadline = time.monotonic() + 60
        while True:
            if process.poll() is not None:
                self.fail("server exited at startup: " + log_path.read_text()[-3000:])
            try:
                if _request(port, "GET", "/health", timeout=1)[0] == 200:
                    if "S41SERVERFFN ready" not in log_path.read_text():
                        self.skipTest("llama-server built without S41_SERVER_FFN_SPLIT")
                    return port, log_path
            except (OSError, http.client.HTTPException, ValueError):
                pass
            if time.monotonic() > deadline:
                process.kill()
                self.fail("server never became ready: " + log_path.read_text()[-3000:])
            time.sleep(0.05)

    def test_new_prompt_runs_while_a_phone_policy_slot_decodes(self) -> None:
        port, log_path = self._server()
        solo = _Stream(port, "anchor-solo", 1, SHORT)
        self.addCleanup(solo.connection.close)
        solo.read()
        self.assertTrue(solo.stopped)

        leader = _Stream(port, "anchor-leader", 0, LONG)
        self.addCleanup(leader.connection.close)
        leader.read(1)
        applied = leader.control(MASK)
        self.assertTrue(applied["success"], applied)
        leader.read(4)
        self.assertIn("FFNCONTROL request=anchor-leader", log_path.read_text())

        joiner = _Stream(port, "anchor-joiner", 1, SHORT)
        self.addCleanup(joiner.connection.close)
        joiner.read()
        _, slots = _request(port, "GET", "/slots")
        lead = next(slot for slot in slots if slot["id"] == 0)
        self.assertTrue(joiner.stopped)
        self.assertTrue(lead["is_processing"],
                        "the joiner finished only after the phone-policy slot (prompt starved by the anchor)")
        self.assertLess(lead["next_token"][0]["n_decoded"] if isinstance(lead["next_token"], list)
                        else lead["next_token"]["n_decoded"], LONG)
        self.assertEqual(joiner.tokens, solo.tokens)

        leader.read()
        self.assertTrue(leader.stopped)
        self.assertEqual(len(leader.tokens), LONG)


if __name__ == "__main__":
    unittest.main()
