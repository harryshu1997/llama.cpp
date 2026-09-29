"""Offline FFN shard GGUF generation and worker shard-mode equivalence."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
sys.path.insert(0, str(REPO_ROOT / "research_dev" / "scheduler" / "native"))

import gguf  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402
from gguf.quants import quantize  # noqa: E402

import ffn_shard_gguf as shard_tool  # noqa: E402

WORKER = REPO_ROOT / "build-cpu" / "bin" / "llama-ffn-split-worker"
N_LAYER = 4
N_EMBD = 64
N_FF = 1024


def _weights(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result = {}
    for layer in range(N_LAYER):
        # numpy shape is (ne1, ne0): gate/up are (n_ff, n_embd), down is (n_embd, n_ff)
        result[f"blk.{layer}.ffn_gate.weight"] = rng.standard_normal((N_FF, N_EMBD)).astype(np.float32)
        result[f"blk.{layer}.ffn_up.weight"] = rng.standard_normal((N_FF, N_EMBD)).astype(np.float32)
        result[f"blk.{layer}.ffn_down.weight"] = rng.standard_normal((N_EMBD, N_FF)).astype(np.float32)
        # unrelated tensors the shard must drop
        result[f"blk.{layer}.attn_q.weight"] = rng.standard_normal((N_EMBD, N_EMBD)).astype(np.float32)
    result["token_embd.weight"] = rng.standard_normal((128, N_EMBD)).astype(np.float32)
    return result


def _write_model(path: Path, weights: dict[str, np.ndarray], qtype: GGMLQuantizationType) -> None:
    writer = gguf.GGUFWriter(str(path), "qwen3")
    writer.add_name("tiny-qwen3-test")
    writer.add_block_count(N_LAYER)
    writer.add_embedding_length(N_EMBD)
    writer.add_feed_forward_length(N_FF)
    for name, data in weights.items():
        if ".ffn_" in name:
            writer.add_tensor(name, quantize(data, qtype), raw_dtype=qtype)
        else:
            writer.add_tensor(name, data.astype(np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_worker(model: Path, artifact: str, layers: str, columns: int, timeout_s: float = 60.0):
    """Start the worker in TCP mode; return (exit_code or None, stderr text)."""
    command = [
        str(WORKER), "-m", str(model), "--artifact-sha256", artifact,
        "--layers", layers, "--columns", str(columns), "--backend", "CPU",
        "--port", str(_free_port()), "--column-quantum", "256",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "S42_RESIDENCY_SESSION_ID": "HTPTEST",
             "S42_RESIDENCY_SESSION_GENERATION": "1"},
    )
    lines = []
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            line = process.stderr.readline()
            if line:
                lines.append(line)
                if "[ffn-worker] ready " in line:
                    break
                continue
            if process.poll() is not None:
                break
            time.sleep(0.01)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        lines.extend(process.stderr.read().splitlines(keepends=True))
        process.stderr.close()
    exit_code = None if any("[ffn-worker] ready " in l for l in lines) else process.returncode
    return exit_code, "".join(lines)


def _ready_fields(stderr: str) -> dict[str, str]:
    match = re.search(r"\[ffn-worker\] ready (.*)$", stderr, re.M)
    if match is None:
        raise AssertionError(f"worker did not become ready:\n{stderr[-2000:]}")
    return dict(re.findall(r"(\w+)=(\S+)", match.group(1)))


class FfnShardGgufTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="s42-ffn-shard-"))
        cls.weights = _weights(7)
        cls.models = {}
        for qtype in (GGMLQuantizationType.F16, GGMLQuantizationType.Q8_0):
            path = cls.root / f"tiny-{qtype.name}.gguf"
            _write_model(path, cls.weights, qtype)
            cls.models[qtype] = (path, _sha256(path))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _shard(self, qtype, requests, name):
        model, sha = self.models[qtype]
        out_dir = self.root / name
        index = shard_tool.write_shards(
            model, sha, out_dir,
            [shard_tool.ShardRequest.parse(r) for r in requests],
            verify_parent=True,
        )
        return model, sha, out_dir, index

    def _reference_slices(self, layer: int, columns: int, qtype):
        """Independent reference: quantize the float slice directly."""
        gate = self.weights[f"blk.{layer}.ffn_gate.weight"][N_FF - columns:, :]
        up = self.weights[f"blk.{layer}.ffn_up.weight"][N_FF - columns:, :]
        down = self.weights[f"blk.{layer}.ffn_down.weight"][:, N_FF - columns:]
        return {
            f"blk.{layer}.ffn_gate.weight": quantize(gate, qtype).view(np.uint8).reshape(-1),
            f"blk.{layer}.ffn_up.weight": quantize(up, qtype).view(np.uint8).reshape(-1),
            f"blk.{layer}.ffn_down.weight": quantize(down, qtype).view(np.uint8).reshape(-1),
        }

    def test_shards_hold_only_selected_ffn_slices(self):
        for qtype in (GGMLQuantizationType.F16, GGMLQuantizationType.Q8_0):
            with self.subTest(qtype=qtype.name):
                model, sha, out_dir, index = self._shard(
                    qtype, ["HTP0=0-1:512", "HTP1=2-3:768"], f"shards-{qtype.name}"
                )
                self.assertEqual(index["parent_sha256"], sha)
                self.assertEqual({s["session_id"] for s in index["shards"]}, {"HTP0", "HTP1"})
                for shard, layers, columns in (
                    (index["shards"][0], (0, 1), 512),
                    (index["shards"][1], (2, 3), 768),
                ):
                    path = out_dir / shard["path"]
                    self.assertEqual(shard["shard_sha256"], _sha256(path))
                    self.assertEqual(shard["shard_bytes"], path.stat().st_size)
                    self.assertEqual(shard["layers"], list(layers))
                    self.assertEqual(shard["column_offset"], N_FF - columns)
                    sidecar = json.loads(path.with_suffix(".json").read_text())
                    self.assertEqual(sidecar["shard_sha256"], shard["shard_sha256"])
                    meta = shard_tool.read_shard_metadata(path)
                    self.assertEqual(meta["version"], 1)
                    self.assertEqual(meta["parent_sha256"], sha)
                    self.assertEqual(meta["n_ff"], N_FF)
                    self.assertEqual(meta["columns"], columns)
                    self.assertEqual(meta["layer_mask"], shard_tool.layer_mask(layers))
                    self.assertEqual(meta["weight_type"], qtype.name)
                    expected_names = {
                        f"blk.{l}.{t}.weight" for l in layers for t in shard_tool.FFN_TENSORS
                    }
                    self.assertEqual(set(meta["tensors"]), expected_names)
                    reader = gguf.GGUFReader(str(path))
                    by_name = {t.name: t for t in reader.tensors}
                    for layer in layers:
                        reference = self._reference_slices(layer, columns, qtype)
                        for name, expected in reference.items():
                            tensor = by_name[name]
                            if name.endswith("ffn_down.weight"):
                                self.assertEqual(tuple(int(x) for x in tensor.shape), (columns, N_EMBD))
                            else:
                                self.assertEqual(tuple(int(x) for x in tensor.shape), (N_EMBD, columns))
                            actual = np.ascontiguousarray(tensor.data).view(np.uint8).reshape(-1)
                            self.assertTrue(np.array_equal(actual, expected), name)
                    # storage shrinks to the selected slices only
                    full = model.stat().st_size
                    self.assertLess(path.stat().st_size, full // 2)

    def test_rejects_misaligned_and_invalid_requests(self):
        model, sha = self.models[GGMLQuantizationType.Q8_0]
        with self.assertRaisesRegex(shard_tool.FfnShardError, "align"):
            shard_tool.write_shards(model, sha, self.root / "bad-align",
                                    [shard_tool.ShardRequest.parse("HTP0=0:500")])
        with self.assertRaisesRegex(shard_tool.FfnShardError, "block_count"):
            shard_tool.write_shards(model, sha, self.root / "bad-layer",
                                    [shard_tool.ShardRequest.parse("HTP0=9:512")])
        with self.assertRaisesRegex(shard_tool.FfnShardError, "duplicate"):
            shard_tool.write_shards(model, sha, self.root / "bad-dup",
                                    [shard_tool.ShardRequest.parse("HTP0=0:512"),
                                     shard_tool.ShardRequest.parse("HTP0=1:512")])
        with self.assertRaisesRegex(shard_tool.FfnShardError, "differs from declared"):
            shard_tool.write_shards(model, "sha256:" + "0" * 64, self.root / "bad-parent",
                                    [shard_tool.ShardRequest.parse("HTP0=0:512")],
                                    verify_parent=True)
        self.assertEqual(shard_tool.layer_spec((0, 1, 2, 5, 7, 8)), "0-2,5,7-8")
        self.assertEqual(shard_tool.parse_layer_spec("0-2,5,7-8"), (0, 1, 2, 5, 7, 8))

    @unittest.skipUnless(WORKER.exists(), "host llama-ffn-split-worker is not built")
    def test_worker_shard_mode_matches_full_model_hash(self):
        # The host CPU worker only uploads f16 FFN weights (the phone shards are
        # f16 as well); Q8_0 byte equivalence is covered by the tool-level test.
        for qtype in (GGMLQuantizationType.F16,):
            with self.subTest(qtype=qtype.name):
                model, sha, out_dir, index = self._shard(
                    qtype, ["HTP0=0-1:512", "HTP1=2-3:768"], f"worker-{qtype.name}"
                )
                shard0 = out_dir / "HTP0.ffn.gguf"
                shard1 = out_dir / "HTP1.ffn.gguf"
                cases = (
                    (shard0, "0-1", 512),   # full stored slice
                    (shard0, "0-1", 256),   # smaller served suffix of the stored slice
                    (shard0, "1", 512),     # layer subset of the stored mask
                    (shard1, "2-3", 768),
                )
                for shard, layers, columns in cases:
                    code_full, err_full = _run_worker(model, sha, layers, columns)
                    code_shard, err_shard = _run_worker(shard, sha, layers, columns)
                    self.assertIsNone(code_full, err_full[-1500:])
                    self.assertIsNone(code_shard, err_shard[-1500:])
                    full = _ready_fields(err_full)
                    sharded = _ready_fields(err_shard)
                    self.assertIn("[ffn-worker] FFN shard parent=", err_shard)
                    for key in ("hash", "mask", "K", "NFF", "slice", "type", "blocks", "weights"):
                        self.assertEqual(full[key], sharded[key], (shard.name, layers, columns, key))
                    self.assertEqual(sharded["NFF"], str(N_FF))
                    self.assertEqual(sharded["slice"], f"[{N_FF - columns},{N_FF})")

    @unittest.skipUnless(WORKER.exists(), "host llama-ffn-split-worker is not built")
    def test_worker_rejects_foreign_or_insufficient_shards(self):
        qtype = GGMLQuantizationType.F16
        model, sha, out_dir, _index = self._shard(qtype, ["HTP0=0-1:512"], "reject")
        shard0 = out_dir / "HTP0.ffn.gguf"
        code, err = _run_worker(shard0, "sha256:" + "1" * 64, "0-1", 512)
        self.assertEqual(code, 1)
        self.assertIn("differs from artifact", err)
        code, err = _run_worker(shard0, sha, "0-2", 512)
        self.assertEqual(code, 1)
        self.assertIn("do not cover", err)
        code, err = _run_worker(shard0, sha, "0-1", 768)
        self.assertEqual(code, 1)
        self.assertIn("stores 512 columns", err)


if __name__ == "__main__":
    unittest.main()
