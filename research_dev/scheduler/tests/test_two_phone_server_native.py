"""Native gate for one llama-server driving two FFN helper workers on disjoint layers.

A tiny llama GGUF (four layers) runs through the real server with local CPU copies of the
phone FFN worker (``llama-ffn-split-worker``, TCP transport):

* ``single`` - the legacy environment, one worker owning layers 0-3;
* ``two``    - ``S41_SERVER_FFN_HELPERS=2``, worker A owns layers 0-1, worker B layers 2-3.

Both arms use the same static column policy and the same column quantum, so every layer is
computed by identical graphs: the generated tokens must be identical, every call must reach
the layer's owner, the two helpers use disjoint request-id ranges and each helper prints its
own summary. A runtime-control arm applies union policies to two deferred helpers, checks
that a policy one helper rejects is refused as a whole, and that startup rejects overlapping,
uncovered or ambiguous helper environments.

Binaries come from ``build-cpu/bin`` (override with ``S42_LLAMA_BUILD_BIN``); the test is
skipped when they are absent.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
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
PHONE_COLUMNS = N_FF // 2
UNION_MASK = 0b1111
CALL = re.compile(r"S41SERVERFFNCALL (?:context=\S+ )?request=(\d+) layer=(\d+) tokens=(\d+) columns=(\d+)")
SECOND_RANGE = 1 + (1 << 24)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _request(port: int, method: str, path: str, body: dict | None = None,
             headers: dict | None = None, timeout: float = 60) -> tuple[int, dict]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method, path, None if body is None else json.dumps(body),
                           {"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"{}")
    finally:
        connection.close()


class TwoHelperServerNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SERVER.exists() or not WORKER.exists():
            raise unittest.SkipTest(f"native binaries missing under {BIN_DIR}")
        cls.tmp = Path(tempfile.mkdtemp(prefix="s42-two-helper-"))
        cls.model = write_tiny_llama_gguf(cls.tmp / "tiny.gguf", n_layer=4, seed=5, with_tokenizer=True)
        cls.parent = _sha256(cls.model)
        shards = cls.tmp / "shards"
        subprocess.run(
            [sys.executable, str(SHARD_TOOL), str(cls.model), "--parent-sha256", cls.parent,
             "--out-dir", str(shards), "--shard", f"HTP0=0-1:{N_FF}", "--shard", f"HTP1=2-3:{N_FF}",
             "--verify-parent"],
            check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "gguf-py")},
        )
        cls.workers = []
        cls.ports = {
            "all": cls._worker("all", cls.model, "0-3", 64),
            "a": cls._worker("a", shards / "HTP0.ffn.gguf", "0-1", 64),
            "b": cls._worker("b", shards / "HTP1.ffn.gguf", "2-3", 64),
            "b128": cls._worker("b128", shards / "HTP1.ffn.gguf", "2-3", 128),
        }

    @classmethod
    def _worker(cls, name: str, model: Path, layers: str, quantum: int) -> int:
        port = _free_port()
        log_path = cls.tmp / f"worker-{name}.log"
        log = log_path.open("w")
        process = subprocess.Popen(
            [str(WORKER), "-m", str(model), "--artifact-sha256", cls.parent, "--layers", layers,
             "--columns", str(N_FF), "--column-quantum", str(quantum), "--backend", "CPU",
             "--port", str(port), "--bind", "127.0.0.1", "--f16-io", "--max-tokens", str(MAX_TOKENS)],
            stdout=log, stderr=subprocess.STDOUT,
        )
        cls.workers.append((process, log))
        deadline = time.monotonic() + 30
        while "ready backend" not in log_path.read_text():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"worker {name} not ready: " + log_path.read_text())
            time.sleep(0.05)
        return port

    @classmethod
    def tearDownClass(cls) -> None:
        for process, log in getattr(cls, "workers", []):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            log.close()

    # ---- helpers ---------------------------------------------------------------------------

    def _environment(self, helpers: dict[str, tuple[str, int]] | None, **shared: str) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if not key.startswith("S41_SERVER_")}
        if helpers is None:
            return env
        env.update({
            "S41_SERVER_FFN_ARTIFACT_SHA256": self.parent,
            "S41_SERVER_FFN_ACTIVATION": "swiglu",
            "S41_SERVER_FFN_COLUMNS": str(N_FF),
            "S41_SERVER_FFN_F16_IO": "1",
            "S41_SERVER_FFN_LAYER_MASK": str(UNION_MASK),
            "S41_SERVER_FFN_N_EMBD": str(N_EMBD),
            "S41_SERVER_FFN_TIMEOUT_MS": "20000",
        })
        if list(helpers) == ["legacy"]:
            env.update({"S41_SERVER_FFN_TRANSPORT": "tcp", "S41_SERVER_FFN_HOST": "127.0.0.1",
                        "S41_SERVER_FFN_PORT": str(self.ports[helpers["legacy"][0]])})
        else:
            env["S41_SERVER_FFN_HELPERS"] = str(len(helpers))
            for index, (label, (worker, mask)) in enumerate(helpers.items()):
                prefix = f"S41_SERVER_FFN_HELPER{index}_"
                env.update({prefix + "LABEL": label, prefix + "LAYER_MASK": str(mask),
                            prefix + "TRANSPORT": "tcp", prefix + "HOST": "127.0.0.1",
                            prefix + "PORT": str(self.ports[worker])})
        env.update(shared)
        return env

    def _start(self, name: str, env: dict[str, str]) -> tuple[subprocess.Popen, Path, int]:
        port = _free_port()
        log_path = self.tmp / f"server-{name}.log"
        command = [str(SERVER), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(port),
                   "--parallel", "1", "--ctx-size", "256", "--batch-size", str(MAX_TOKENS),
                   "--ubatch-size", str(MAX_TOKENS), "--n-gpu-layers", "0", "--threads", "2", "--no-warmup"]
        with log_path.open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 60
        while True:
            if process.poll() is not None:
                return process, log_path, port
            try:
                if _request(port, "GET", "/health", timeout=1)[0] == 200:
                    return process, log_path, port
            except (OSError, http.client.HTTPException, ValueError):
                pass
            if time.monotonic() > deadline:
                process.kill()
                self.fail(f"server {name} never became ready: " + log_path.read_text()[-3000:])
            time.sleep(0.05)

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)

    def _complete(self, port: int, n_predict: int = 12) -> list[int]:
        status, body = _request(port, "POST", "/completion", {
            "prompt": PROMPT, "n_predict": n_predict, "temperature": 0, "seed": 7, "ignore_eos": True,
            "return_tokens": True, "cache_prompt": False, "id_slot": 0})
        self.assertEqual(status, 200, body)
        return body["tokens"]

    def _static_arm(self, name: str, helpers: dict[str, tuple[str, int]] | None) -> tuple[list[int], str]:
        shared = {} if helpers is None else {"S41_SERVER_FFN_POLICY": f"{MAX_TOKENS}:{PHONE_COLUMNS}"}
        process, log_path, port = self._start(name, self._environment(helpers, **shared))
        try:
            self.assertIsNone(process.poll(), log_path.read_text()[-3000:])
            tokens = self._complete(port)
        finally:
            self._stop(process)
        return tokens, log_path.read_text()

    # ---- tests -----------------------------------------------------------------------------

    def test_two_helpers_match_one_helper_and_split_calls_by_owner(self) -> None:
        host_tokens, _ = self._static_arm("host", None)
        single_tokens, single_log = self._static_arm("single", {"legacy": ("all", UNION_MASK)})
        two_tokens, two_log = self._static_arm("two", {"op15": ("a", 0b0011), "pixel": ("b", 0b1100)})
        self.assertEqual(len(host_tokens), 12)
        self.assertEqual(two_tokens, single_tokens)

        single_calls = [tuple(map(int, row)) for row in CALL.findall(single_log)]
        two_calls = [tuple(map(int, row)) for row in CALL.findall(two_log)]
        self.assertTrue(single_calls)
        per_layer = lambda calls: sorted((layer, tokens, columns) for _, layer, tokens, columns in calls)  # noqa: E731
        self.assertEqual(per_layer(two_calls), per_layer(single_calls))
        self.assertTrue(all(columns == PHONE_COLUMNS for *_, columns in two_calls))
        first = [request for request, layer, *_ in two_calls if layer in (0, 1)]
        second = [request for request, layer, *_ in two_calls if layer in (2, 3)]
        self.assertTrue(first and second)
        self.assertTrue(all(request < SECOND_RANGE for request in first))
        self.assertTrue(all(request >= SECOND_RANGE for request in second))
        self.assertEqual(len(set(first + second)), len(two_calls))
        self.assertEqual(sorted(second), list(range(SECOND_RANGE, SECOND_RANGE + len(second))))

        self.assertIn("S41SERVERFFNHELPER label=op15 layer_mask=3 transport=tcp", two_log)
        self.assertIn("S41SERVERFFNHELPER label=pixel layer_mask=12 transport=tcp", two_log)
        self.assertEqual(len(re.findall(r"^S41SERVERFFN ready ", two_log, re.M)), 1)
        summaries = [json.loads(line.split(" ", 1)[1]) for line in two_log.splitlines()
                     if line.startswith("S41SERVERFFN {")]
        self.assertEqual({row["helper"]: row["layer_mask"] for row in summaries}, {"op15": 3, "pixel": 12})
        self.assertEqual(sum(row["calls"] for row in summaries), len(two_calls))
        self.assertTrue(all(row["status"] == "ok" for row in summaries))
        legacy = [line for line in single_log.splitlines() if line.startswith("S41SERVERFFN {")]
        self.assertEqual(len(legacy), 1)
        self.assertTrue(legacy[0].startswith('S41SERVERFFN {"status":"ok"'))
        self.assertNotIn("S41SERVERFFNHELPER", single_log)

    def test_runtime_control_applies_union_policy_and_refuses_partial_acceptance(self) -> None:
        env = self._environment({"op15": ("a", 0b0011), "pixel": ("b128", 0b1100)},
                                S41_SERVER_FFN_RUNTIME_CONTROL="1",
                                S41_SERVER_FFN_MAX_TOKENS=str(MAX_TOKENS))
        process, log_path, port = self._start("runtime", env)
        try:
            self.assertIsNone(process.poll(), log_path.read_text()[-3000:])
            text = log_path.read_text()
            self.assertIn("connection=deferred helpers=2", text)
            self.assertIn("S41SERVERFFNHELPER label=pixel layer_mask=12 transport=tcp", text)
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            connection.request("POST", "/completion", json.dumps({
                "prompt": PROMPT, "n_predict": 200, "temperature": 0, "ignore_eos": True, "stream": True,
                "cache_prompt": False, "id_slot": 0}),
                {"Content-Type": "application/json", "X-Scheduler-Request-ID": "two-phone-runtime"})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            while not response.readline().startswith(b"data: "):
                pass

            def control(generation: int, columns: int, layer_mask: int = UNION_MASK) -> tuple[int, dict]:
                return _request(port, "POST", "/v1/chat/completions/control", {
                    "action": "ffn_split", "request_id": "two-phone-runtime", "slot_id": 0,
                    "plan_generation": generation, "policy_hash": "sha256:" + f"{generation:064x}",
                    "layer_mask": layer_mask, "columns": columns, "enabled": columns != 0})

            try:
                # quantum 64 on op15 and 128 on pixel: 64 columns is accepted by one helper only
                status, body = control(1, 64)
                self.assertEqual((status, body.get("success")), (200, False), body)
                self.assertIn("helper pixel:", body["message"])
                status, body = control(2, 128)
                self.assertEqual((status, body.get("success")), (200, True), body)
                status, body = control(3, 128, layer_mask=0b0110)
                self.assertEqual((status, body.get("success")), (200, True), body)
                while response.readline():
                    pass
            finally:
                connection.close()
        finally:
            self._stop(process)
        log = log_path.read_text()
        calls = [tuple(map(int, row)) for row in CALL.findall(log)]
        layers = {layer for _, layer, *_ in calls}
        self.assertEqual(layers, {0, 1, 2, 3})
        self.assertTrue(all(columns == 128 for *_, columns in calls))
        self.assertTrue(all((request >= SECOND_RANGE) == (layer >= 2) for request, layer, *_ in calls))
        summaries = {json.loads(line.split(" ", 1)[1])["helper"]
                     for line in log.splitlines() if line.startswith("S41SERVERFFN {")}
        self.assertEqual(summaries, {"op15", "pixel"})
        self.assertNotIn("S41SERVERFFNERROR", log)

    def test_startup_rejects_ambiguous_helper_environments(self) -> None:
        good = {"op15": ("a", 0b0011), "pixel": ("b", 0b1100)}
        cases = {
            "overlap": ({"op15": ("a", 0b0111), "pixel": ("b", 0b1100)}, {},
                        "must be nonempty, disjoint subsets"),
            "uncovered": ({"op15": ("a", 0b0001), "pixel": ("b", 0b1100)}, {}, "do not cover"),
            "legacy-next-to-helpers": (good, {"S41_SERVER_FFN_HOST": "127.0.0.1"}, "legacy transport"),
            "two-functionfs": (good, {
                "S41_SERVER_FFN_HELPER0_TRANSPORT": "functionfs-usb",
                "S41_SERVER_FFN_HELPER1_TRANSPORT": "functionfs-usb",
                **{f"S41_SERVER_FFN_HELPER{index}_{key}": value for index in (0, 1) for key, value in (
                    ("USB_ALLOCATOR", "malloc"), ("USB_TRANSPORT_GENERATION", "g"),
                    ("USB_BATCH_PLAN", "split-row"), ("USB_QUEUE_DEPTH", "1"),
                    ("USBFS_AVAILABLE_BYTES", "0"), ("USB_SLOT_SAFETY_BYTES", "65536"),
                    ("USB_MAX_PAYLOAD_BYTES", "4096"), ("USB_VENDOR_ID", "6353"),
                    ("USB_PRODUCT_ID", "11520"), ("USB_SPLIT_H2D", "0"), ("USB_FULL_DUPLEX", "0"))},
            }, "at most one server FFN helper can use the functionfs-usb transport"),
            "duplicate-label": (good, {"S41_SERVER_FFN_HELPER1_LABEL": "op15"}, "duplicate server FFN helper label"),
        }
        for name, (helpers, extra, message) in cases.items():
            with self.subTest(case=name):
                env = self._environment(helpers, S41_SERVER_FFN_RUNTIME_CONTROL="1", **extra)
                process, log_path, _ = self._start("reject-" + name, env)
                try:
                    process.wait(timeout=60)
                finally:
                    self._stop(process)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn(message, log_path.read_text())


if __name__ == "__main__":
    unittest.main()
