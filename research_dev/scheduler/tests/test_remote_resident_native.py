"""Native gate for remote-resident FFN weights (first milestone, software validation).

Runs the real llama.cpp loader and graph on a tiny llama-architecture GGUF in two arms:

* ``full``   - every weight local, no FFN client;
* ``remote`` - the gate/up/down weights of the masked layers are omitted from the desktop
  process and executed by ``llama-ffn-split-worker`` (the phone worker binary, host build)
  over TCP through the server's FFN split client.

Checks (loader, identity, execution, controls groups of the milestone plan):

* allocation proof: the loader reports the omitted bytes, releases the page-exact interior of
  every omitted tensor, and ``/proc/self/smaps`` shows no VMA of the model file over those
  pages (an independent kernel-level record, not a view or a smaller virtual span);
* the file mapping shrinks by at least the released bytes relative to the full arm;
* execution: prefill plus greedy decode produce identical argmax traces and logits within
  ``LOGITS_ABS_TOLERANCE`` (f16 weights, f32 activations in both arms);
* fail closed: a context without an eval-callback owner is refused, and a runtime control
  that targets a remote-resident layer is rejected by the client.

The binaries are looked up in ``build-cpu/bin`` (override with ``S42_LLAMA_BUILD_BIN``); the
test is skipped when they are absent so the canonical suite stays runnable everywhere.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
SCHEDULER_DIR = TESTS_DIR.parent
REPO_ROOT = SCHEDULER_DIR.parents[1]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
if str(REPO_ROOT / "gguf-py") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from tiny_llama_gguf import N_EMBD, N_FF, write_tiny_llama_gguf  # noqa: E402

BIN_DIR = Path(os.environ.get("S42_LLAMA_BUILD_BIN", REPO_ROOT / "build-cpu" / "bin"))
PROBE = BIN_DIR / "llama-ffn-remote-resident-probe"
WORKER = BIN_DIR / "llama-ffn-split-worker"
SHARD_TOOL = SCHEDULER_DIR / "native" / "ffn_shard_gguf.py"

REMOTE_LAYERS = (1, 2)
REMOTE_MASK = sum(1 << layer for layer in REMOTE_LAYERS)
MAX_TOKENS = 16
PROMPT = (1, 2, 3, 4, 5, 6, 7, 8)
DECODE_STEPS = 4
LOGITS_ABS_TOLERANCE = 1e-4
PAGE = os.sysconf("SC_PAGE_SIZE")


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


class RemoteResidentNativeTests(unittest.TestCase):
    """Loader omission, kernel proof, numerical identity and fail-closed controls."""

    @classmethod
    def setUpClass(cls) -> None:
        if not PROBE.exists() or not WORKER.exists():
            raise unittest.SkipTest(f"native binaries missing under {BIN_DIR}")
        cls.tmp = Path(tempfile.mkdtemp(prefix="s42-remote-resident-"))
        cls.model = write_tiny_llama_gguf(cls.tmp / "tiny_llama.gguf", n_layer=4)
        cls.parent_sha256 = _sha256(cls.model)
        shard_dir = cls.tmp / "shards"
        layer_spec = f"{REMOTE_LAYERS[0]}-{REMOTE_LAYERS[-1]}"
        subprocess.run(
            [sys.executable, str(SHARD_TOOL), str(cls.model),
             "--parent-sha256", cls.parent_sha256, "--out-dir", str(shard_dir),
             "--shard", f"HTP0={layer_spec}:{N_FF}", "--verify-parent"],
            check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "gguf-py")},
        )
        cls.shard_index = json.loads((shard_dir / "FFN_SHARDS.json").read_text())
        cls.port = _free_port()
        cls.worker_log = (cls.tmp / "worker.log").open("w")
        cls.worker = subprocess.Popen(
            [str(WORKER), "-m", str(shard_dir / "HTP0.ffn.gguf"),
             "--artifact-sha256", cls.parent_sha256, "--layers", layer_spec,
             "--columns", str(N_FF), "--backend", "CPU", "--port", str(cls.port),
             "--max-tokens", str(MAX_TOKENS)],
            stdout=cls.worker_log, stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if "ready backend" in (cls.tmp / "worker.log").read_text():
                break
            if cls.worker.poll() is not None:
                raise RuntimeError("FFN worker exited: " + (cls.tmp / "worker.log").read_text())
            time.sleep(0.1)
        else:
            raise RuntimeError("FFN worker did not become ready")
        cls.full = cls._probe("full", [])
        cls.remote = cls._probe("remote", [
            "--remote-mask", str(REMOTE_MASK), "--worker-port", str(cls.port),
            "--artifact-sha256", cls.parent_sha256, "--max-tokens", str(MAX_TOKENS),
        ])
        cls.no_owner = cls._probe("noowner", [
            "--remote-mask", str(REMOTE_MASK), "--expect-no-owner",
        ], logits=False)

    @classmethod
    def tearDownClass(cls) -> None:
        worker = getattr(cls, "worker", None)
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
        log = getattr(cls, "worker_log", None)
        if log is not None:
            log.close()
        if os.environ.get("S42_KEEP_NATIVE_ARTIFACTS") != "1" and hasattr(cls, "tmp"):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def _probe(cls, name: str, extra: list[str], *, logits: bool = True) -> dict:
        out = cls.tmp / f"{name}.json"
        command = [str(PROBE), "--model", str(cls.model), "--out", str(out),
                   "--tokens", ",".join(str(token) for token in PROMPT),
                   "--decode", str(DECODE_STEPS), *extra]
        if logits:
            command += ["--logits", str(cls.tmp / f"{name}.bin")]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if completed.returncode != 0 or not out.exists():
            raise RuntimeError(f"probe {name} failed rc={completed.returncode}: {completed.stderr[-2000:]}")
        record = json.loads(out.read_text())
        record["stderr"] = completed.stderr
        if logits:
            record["logits"] = np.fromfile(cls.tmp / f"{name}.bin", dtype=np.float32)
        return record

    # ---- loader group -------------------------------------------------------------------

    def test_loader_reports_exact_omission(self) -> None:
        loader = self.remote["loader"]
        per_tensor = N_FF * N_EMBD * 2  # f16
        self.assertEqual(loader["remote_mask"], REMOTE_MASK)
        self.assertEqual(loader["omitted_bytes"], 3 * len(REMOTE_LAYERS) * per_tensor)
        self.assertEqual(loader["model_size_bytes"], self.full["loader"]["model_size_bytes"],
                         "the artifact size is a fact about the file and must not change")
        expected_unmapped = sum(
            max(0, row["page_last"] - row["page_first"]) for row in self.remote["omitted_tensors"]
        )
        self.assertEqual(loader["unmapped_bytes"], expected_unmapped)
        self.assertGreater(loader["unmapped_bytes"], 0)
        self.assertLessEqual(loader["unmapped_bytes"], loader["omitted_bytes"])
        self.assertEqual(self.full["loader"]["omitted_bytes"], 0)
        self.assertEqual(self.full["loader"]["unmapped_bytes"], 0)

    def test_kernel_mapping_has_no_vma_over_omitted_pages(self) -> None:
        rows = self.remote["omitted_tensors"]
        self.assertEqual(len(rows), 3 * len(REMOTE_LAYERS))
        for row in rows:
            with self.subTest(tensor=row["name"]):
                self.assertGreater(row["page_last"], row["page_first"],
                                   "fixture tensors must span whole pages for the proof to bite")
                self.assertEqual(row["vma_overlap_bytes"], 0)
                self.assertEqual(row["rss_overlap_bytes"], 0)
        full_mapped = self.full["file_mapping"]["mapped_bytes"]
        remote_mapped = self.remote["file_mapping"]["mapped_bytes"]
        self.assertGreaterEqual(full_mapped - remote_mapped, self.remote["loader"]["unmapped_bytes"])
        self.assertGreater(self.remote["file_mapping"]["vma_count"], self.full["file_mapping"]["vma_count"],
                           "interior holes split the single file mapping into several VMAs")
        self.assertTrue(any("REMOTE_RESIDENT_FFN" in line for line in self.remote["log"]))

    # ---- execution / identity group -------------------------------------------------------

    def test_server_raw_logits_capture_preserves_two_slot_outputs(self) -> None:
        server = BIN_DIR / 'llama-server'
        if not server.exists():
            self.skipTest('llama-server is absent from this build')
        model = write_tiny_llama_gguf(self.tmp / 'tiny_server.gguf', with_tokenizer=True)
        reference = None
        for enabled in (False, True):
            port = _free_port()
            command = [str(server), '-m', str(model), '--host', '127.0.0.1', '--port', str(port),
                       '--parallel', '2', '--ctx-size', '1024', '--batch-size', '16', '--ubatch-size', '16',
                       '--n-gpu-layers', '0', '--threads', '2', '--no-warmup']
            env = {key: value for key, value in os.environ.items() if not key.startswith('S41_SERVER_')}
            trace = self.tmp / 'server-raw-logits.bin'
            if enabled:
                env['S41_SERVER_LOGITS_TRACE'] = str(trace)
            log_path = self.tmp / f'server-raw-logits-{enabled}.log'
            with log_path.open('w') as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        self.assertIsNone(process.poll(), log_path.read_text()[-3000:])
                        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                        try:
                            connection.request('GET', '/health')
                            if connection.getresponse().status == 200:
                                break
                        except (OSError, http.client.HTTPException):
                            pass
                        finally:
                            connection.close()
                        time.sleep(0.05)
                    else:
                        self.fail('tiny logits server never became ready')
                    self.assertEqual(Path(f'/proc/{process.pid}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0'), command)
                    self.assertIn(f'pid={process.pid},', subprocess.check_output(
                        ['ss', '-ltnp', f'sport = :{port}'], text=True))

                    def complete(slot):
                        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
                        try:
                            connection.request('POST', '/completion', json.dumps({
                                'id_slot': slot, 'prompt': list(PROMPT) + [slot + 9], 'n_predict': 8,
                                'temperature': 0, 'seed': 17, 'ignore_eos': True,
                                'return_tokens': True, 'cache_prompt': False}), {'Content-Type': 'application/json'})
                            response = connection.getresponse()
                            body = json.loads(response.read())
                            self.assertEqual(response.status, 200, body)
                            self.assertEqual(len(body['tokens']), 8)
                            return body
                        finally:
                            connection.close()

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        outputs = list(executor.map(complete, (0, 1)))
                    tokens = [row['tokens'] for row in outputs]
                    if not enabled:
                        self.assertFalse(trace.exists())
                        reference = tokens
                    else:
                        self.assertEqual(tokens, reference)
                        self.assertIn('S41SERVERLOGITS schema=s41-logits-v1', log_path.read_text())
                        rows = {}
                        with trace.open('rb') as stream:
                            self.assertEqual(stream.read(8), b'S41LOG1\0')
                            while header := stream.read(16):
                                slot, task, step, count = struct.unpack('<IIII', header)
                                values = np.fromfile(stream, dtype='<f4', count=count)
                                self.assertEqual(len(values), count)
                                self.assertEqual(count, 128)
                                self.assertNotIn((slot, step), rows)
                                self.assertTrue(np.isfinite(values).all())
                                rows[slot, step] = int(np.argmax(values))
                        self.assertEqual(rows, {(slot, step + 1): token for slot, sequence in enumerate(tokens)
                                                for step, token in enumerate(sequence)})
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)

    def test_server_cohort_route_accepts_eight_members_and_rejects_nine(self) -> None:
        server = BIN_DIR / "llama-server"
        if not server.exists():
            self.skipTest("llama-server is absent from this build")
        port = _free_port()
        command = [str(server), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(port),
                   "--parallel", "8", "--ctx-size", "2048", "--batch-size", "16", "--ubatch-size", "16",
                   "--n-gpu-layers", "0", "--threads", "2", "--no-warmup"]
        env = {key: value for key, value in os.environ.items() if not key.startswith("S41_SERVER_FFN_")}
        with (self.tmp / "server-cohort.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    self.assertIsNone(process.poll(), (self.tmp / "server-cohort.log").read_text()[-3000:])
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                    try:
                        connection.request("GET", "/health")
                        if connection.getresponse().status == 200:
                            break
                    except (OSError, http.client.HTTPException):
                        pass
                    finally:
                        connection.close()
                    time.sleep(0.05)
                else:
                    self.fail("tiny server never became ready")
                self.assertEqual(Path(f"/proc/{process.pid}/cmdline").read_bytes().rstrip(b"\0").decode().split("\0"), command)
                listeners = subprocess.check_output(["ss", "-ltnp", f"sport = :{port}"], text=True)
                self.assertIn(f"pid={process.pid},", listeners)
                for size, status in ((8, 200), (9, 400)):
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    try:
                        connection.request("POST", "/v1/chat/completions/control", json.dumps({
                            "action": "ffn_split_cohort_stats", "members": [
                                {"request_id": f"request-{index}", "slot_id": index} for index in range(size)]}),
                            {"Content-Type": "application/json"})
                        response = connection.getresponse()
                        body = json.loads(response.read())
                        self.assertEqual(response.status, status, body)
                        if size == 8:
                            self.assertEqual(body["message"], "FFN cohort members and active slots differ")
                    finally:
                        connection.close()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)

    def test_microbatch_context_tracks_rows_and_request_generations(self) -> None:
        prompt = list(range(1, 38))
        for sequences in (1, 2, 4, 8):
            with self.subTest(sequences=sequences):
                shape = ["--tokens", ",".join(map(str, prompt)), "--batch-size", "64",
                         "--ubatch-size", "8", "--sequence-count", str(sequences),
                         "--runtime-context", "ubatch"]
                full = self._probe(f"micro-full-{sequences}", shape)
                remote = self._probe(f"micro-remote-{sequences}", [*shape,
                    "--remote-mask", str(REMOTE_MASK), "--worker-port", str(self.port),
                    "--artifact-sha256", self.parent_sha256])
                self.assertEqual(remote["decode"]["status"], 0, remote["decode"]["error"])
                self.assertEqual(remote["decode"]["argmax"], full["decode"]["argmax"])
                self.assertLessEqual(float(np.max(np.abs(remote["logits"] - full["logits"]))),
                                     LOGITS_ABS_TOLERANCE)
                counts = [0] * sequences
                for batch in remote["runtime_batches"]:
                    self.assertLessEqual(sum(row["rows"] for row in batch), 8)
                    for row in batch:
                        slot = row["slot_id"]
                        self.assertEqual(row["request_id"], f"probe-{slot}")
                        self.assertEqual(row["plan_generation"], 11 + slot)
                        counts[slot] += row["rows"]
                expected = [len(prompt[slot::sequences]) for slot in range(sequences)]
                expected[0] += DECODE_STEPS
                self.assertEqual(counts, expected)
                self.assertEqual(remote["request_rows"], [n * len(REMOTE_LAYERS) for n in expected])
                calls = [line for line in remote["stderr"].splitlines()
                         if line.startswith("S41SERVERFFNCALL")]
                self.assertEqual(len(calls), len(remote["runtime_batches"]) * len(REMOTE_LAYERS))

    def test_assisted_row_diagnostic_has_five_steps_and_preserves_logits(self) -> None:
        for steps in (5, 64):
            extra = ["--assisted-mask", str(REMOTE_MASK), "--worker-port", str(self.port),
                     "--artifact-sha256", self.parent_sha256, "--runtime-context", "ubatch", "--decode", str(steps + 2)]
            reference = self._probe(f"diagnostic-reference-{steps}", ["--decode", str(steps + 2)])
            ordinary = self._probe(f"diagnostic-disabled-{steps}", extra)
            with patch.dict(os.environ, {"S41_SERVER_FFN_ROW_DIAGNOSTIC_STEPS": str(steps)}):
                diagnostic = self._probe(f"diagnostic-enabled-{steps}", extra)
            for result in (ordinary, diagnostic):
                self.assertEqual(result["decode"]["status"], 0, result["decode"]["error"])
                self.assertEqual(result["decode"]["argmax"], reference["decode"]["argmax"])
                self.assertLessEqual(float(np.max(np.abs(result["logits"] - reference["logits"]))),
                                     LOGITS_ABS_TOLERANCE)
            self.assertNotIn("S41SERVERFFNROW", ordinary["stderr"])
            rows = [dict(field.split("=", 1) for field in line.split()[1:])
                    for line in diagnostic["stderr"].splitlines() if line.startswith("S41SERVERFFNROW ")]
            self.assertEqual(len(rows), steps * len(REMOTE_LAYERS) * 3)
            self.assertEqual({int(row["step"]) for row in rows}, set(range(1, steps + 1)))
            groups = {}
            for row in rows:
                self.assertEqual(row["ubatch_row"], row["payload_row"])
                self.assertEqual(row["slot_id"], "0")
                self.assertEqual(len(row["f32_sha256"]), 64)
                groups.setdefault((row["call"], row["ubatch_row"]), {})[row["stage"]] = row
            for group in groups.values():
                self.assertEqual(set(group), {"input", "local", "returned"})
                self.assertEqual(group["local"]["wire_sha256"], group["returned"]["wire_sha256"])
                # numeric distance of the returned row to the local shadow: exact worker -> zero, with a
                # non-zero reference norm; the input/local stages carry the -1 sentinel
                self.assertEqual(float(group["returned"]["local_rel_l2"]), 0.0)
                self.assertEqual(float(group["returned"]["local_max_abs"]), 0.0)
                self.assertGreater(float(group["returned"]["local_l2"]), 0.0)
                self.assertEqual(float(group["local"]["local_rel_l2"]), -1.0)
                self.assertEqual(float(group["input"]["local_rel_l2"]), -1.0)

    def test_logical_batch_context_still_fails_closed(self) -> None:
        result = self._probe("logical-context-rejected", [
            "--tokens", ",".join(map(str, range(1, 38))), "--batch-size", "64",
            "--ubatch-size", "8", "--runtime-context", "logical",
            "--remote-mask", str(REMOTE_MASK), "--worker-port", str(self.port),
            "--artifact-sha256", self.parent_sha256], logits=False)
        self.assertNotEqual(result["decode"]["status"], 0)
        self.assertIn("runtime context differs from tensor rows", result["decode"]["error"])
        self.assertEqual(result["request_rows"], [0])
        self.assertNotIn("S41SERVERFFNCALL", result["stderr"])

    def test_rejected_microbatch_context_executes_no_phone_calls(self) -> None:
        result = self._probe("micro-context-rejected", [
            "--runtime-context", "reject", "--remote-mask", str(REMOTE_MASK),
            "--worker-port", str(self.port), "--artifact-sha256", self.parent_sha256], logits=False)
        self.assertEqual(result["decode"]["status"], 2)
        self.assertEqual(result["runtime_batches"], [])
        self.assertEqual(result["request_rows"], [0])
        self.assertNotIn("S41SERVERFFNCALL", result["stderr"])

    def test_remote_execution_matches_full_weights(self) -> None:
        self.assertEqual(self.full["decode"]["status"], 0, self.full["decode"]["error"])
        self.assertEqual(self.remote["decode"]["status"], 0, self.remote["decode"]["error"])
        self.assertEqual(self.remote["decode"]["error"], "")
        self.assertEqual(self.remote["decode"]["argmax"], self.full["decode"]["argmax"])
        self.assertEqual(len(self.full["decode"]["argmax"]), DECODE_STEPS + 1)
        full_logits = self.full["logits"]
        remote_logits = self.remote["logits"]
        self.assertEqual(full_logits.shape, remote_logits.shape)
        self.assertLessEqual(float(np.max(np.abs(full_logits - remote_logits))), LOGITS_ABS_TOLERANCE)

    def test_every_remote_layer_call_goes_to_the_phone_at_full_width(self) -> None:
        calls = [line for line in self.remote["stderr"].splitlines() if line.startswith("S41SERVERFFNCALL")]
        # prefill (one call per remote layer) + one call per remote layer per decode step
        self.assertEqual(len(calls), len(REMOTE_LAYERS) * (1 + DECODE_STEPS))
        layers = []
        for line in calls:
            fields = dict(item.split("=", 1) for item in line.split()[1:])
            layers.append(int(fields["layer"]))
            self.assertEqual(int(fields["columns"]), N_FF)
        self.assertEqual(set(layers), set(REMOTE_LAYERS))
        self.assertEqual(int(dict(item.split("=", 1) for item in calls[0].split()[1:])["tokens"]), len(PROMPT))

    # ---- controls / fail-closed group -----------------------------------------------------

    def test_context_without_owner_is_refused(self) -> None:
        self.assertFalse(self.no_owner["context_created"])
        self.assertEqual(self.no_owner["loader"]["remote_mask"], REMOTE_MASK)
        self.assertTrue(any("no FFN eval callback owner" in line for line in self.no_owner["log"]))

    def test_control_targeting_remote_layer_is_rejected(self) -> None:
        self.assertTrue(self.remote["client"]["connected"], self.remote["client"]["error"])
        self.assertEqual(self.remote["client"]["remote_policy_rejection"],
                         "FFN split runtime policy targets remote-resident layers")

    def test_shard_is_a_complete_group_with_parent_identity(self) -> None:
        shard = self.shard_index["shards"][0]
        self.assertEqual(shard["parent_sha256"], self.parent_sha256)
        self.assertEqual(shard["column_offset"], 0)
        self.assertEqual(shard["columns"], N_FF)
        self.assertEqual(int(shard["layer_mask"], 16), REMOTE_MASK)
        names = {row["name"] for row in shard["tensors"]}
        for layer in REMOTE_LAYERS:
            for kind in ("ffn_gate", "ffn_up", "ffn_down"):
                self.assertIn(f"blk.{layer}.{kind}.weight", names)


if __name__ == "__main__":
    unittest.main()


class DormantHostShareNativeTests(unittest.TestCase):
    """Decode-only relocation primitive: release the phone-executed FFN column suffix of mapped weights.

    The tiny fixture keeps every weight memory-mapped on the CPU. After a local decode the probe
    releases the suffix [host_columns, n_ff) of the masked layers' gate/up/down tensors, decodes
    the same prompt again from a fresh context (the pages fault back in from the file), then
    populates the pages. The kernel's own accounting (/proc/self/smaps) is the proof.
    """

    LAYER_MASK = 0b0110
    HOST_COLUMNS = N_FF // 2

    @classmethod
    def setUpClass(cls) -> None:
        if not PROBE.exists():
            raise unittest.SkipTest(f"native probe missing: {PROBE}")
        cls.tmp = Path(tempfile.mkdtemp(prefix="s42-dormant-"))
        cls.model = write_tiny_llama_gguf(cls.tmp / "tiny.gguf", n_layer=4, seed=3)
        out = cls.tmp / "dormant.json"
        command = [str(PROBE), "--model", str(cls.model), "--out", str(out),
                   "--tokens", ",".join(str(token) for token in PROMPT), "--decode", str(DECODE_STEPS),
                   "--dormant-mask", str(cls.LAYER_MASK), "--dormant-host-columns", str(cls.HOST_COLUMNS),
                   "--logits", str(cls.tmp / "dormant.bin")]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if completed.returncode != 0 or not out.exists():
            raise RuntimeError(f"dormant probe failed rc={completed.returncode}: {completed.stderr[-2000:]}")
        cls.record = json.loads(out.read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        if os.environ.get("S42_KEEP_NATIVE_ARTIFACTS") != "1" and hasattr(cls, "tmp"):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def _expected_release_bytes(cls) -> tuple[int, int]:
        """Page-inward bytes of the phone suffix per the GGUF tensor layout: (gate+up, down)."""
        import gguf

        reader = gguf.GGUFReader(str(cls.model))
        gate_up = down = 0
        for tensor in reader.tensors:
            name = tensor.name
            if not name.startswith("blk."):
                continue
            layer = int(name.split(".")[1])
            if not cls.LAYER_MASK >> layer & 1:
                continue
            offs = int(tensor.data_offset)
            element = 2  # f16 weights in the fixture

            def inward(first: int, last: int) -> int:
                first = (first + PAGE - 1) // PAGE * PAGE
                last = last // PAGE * PAGE
                return max(0, last - first)

            if name.endswith((".ffn_gate.weight", ".ffn_up.weight")):
                n_embd, n_ff = int(tensor.shape[0]), int(tensor.shape[1])
                self_check = n_embd == N_EMBD and n_ff == N_FF
                assert self_check, (name, tensor.shape)
                gate_up += inward(offs + cls.HOST_COLUMNS * n_embd * element, offs + n_ff * n_embd * element)
            elif name.endswith(".ffn_down.weight"):
                n_ff, n_embd = int(tensor.shape[0]), int(tensor.shape[1])
                assert n_ff == N_FF and n_embd == N_EMBD, (name, tensor.shape)
                row = n_ff * element
                for r in range(n_embd):
                    down += inward(offs + r * row + cls.HOST_COLUMNS * element, offs + r * row + n_ff * element)
        return gate_up, down

    def test_release_matches_the_page_exact_suffix_geometry(self) -> None:
        dormant = self.record["dormant"]
        self.assertTrue(dormant["ran"])
        gate_up, down = self._expected_release_bytes()
        self.assertGreater(gate_up, 0)
        self.assertEqual(dormant["released_bytes"], gate_up + down)
        self.assertEqual(dormant["range_count"], 2 * 2 + 2 * N_EMBD)  # gate+up per layer, one range per down row
        self.assertEqual(dormant["restored_bytes"], dormant["released_bytes"])

    def test_kernel_accounting_drops_and_recovers_the_released_pages(self) -> None:
        dormant = self.record["dormant"]
        self.assertEqual(dormant["vma_count_before"], dormant["vma_count_after"], "release must not unmap")
        drop = dormant["rss_before"] - dormant["rss_after"]
        self.assertGreaterEqual(drop, dormant["released_bytes"] - PAGE, dormant)
        self.assertGreaterEqual(dormant["rss_restored"], dormant["rss_before"] - PAGE, dormant)

    def test_local_execution_after_release_is_bit_identical(self) -> None:
        dormant = self.record["dormant"]
        self.assertEqual(dormant["decode_status"], 0)
        self.assertTrue(dormant["logits_identical"], dormant)
        self.assertEqual(dormant["max_abs_diff"], 0.0)

    def test_cache_and_restore_policies_preserve_local_logits(self) -> None:
        for drop_cache, populate in ((1, 1), (0, 1), (0, 0), (1, 0)):
            with self.subTest(drop_cache=drop_cache, populate=populate):
                out = self.tmp / f"policy-{drop_cache}-{populate}.json"
                command = [str(PROBE), "--model", str(self.model), "--out", str(out),
                           "--tokens", ",".join(map(str, PROMPT)), "--decode", str(DECODE_STEPS),
                           "--dormant-mask", str(self.LAYER_MASK), "--dormant-host-columns", str(self.HOST_COLUMNS),
                           "--dormant-drop-cache", str(drop_cache), "--dormant-populate", str(populate),
                           "--dormant-restore-before-decode"]
                completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
                self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
                record = json.loads(out.read_text())
                dormant = record["dormant"]
                self.assertEqual(record["decode"]["argmax"], self.record["decode"]["argmax"])
                self.assertTrue(dormant["argmax_identical"], dormant)
                self.assertTrue(dormant["logits_identical"], dormant)
                self.assertEqual(dormant["restored_bytes"], dormant["released_bytes"])
                self.assertEqual((dormant["drop_cache"], dormant["populate"]), (bool(drop_cache), bool(populate)))
                if not populate:
                    self.assertLessEqual(dormant["rss_restored"], dormant["rss_after"] + PAGE)
