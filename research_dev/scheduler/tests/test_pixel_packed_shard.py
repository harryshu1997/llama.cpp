"""WS11 Pixel packed shards for any layer set: byte copy of the quantized origin, exact-dequant check, metadata."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
sys.path.insert(0, str(REPO_ROOT / "research_dev" / "scheduler" / "native"))

import gguf  # noqa: E402
from gguf.constants import GGMLQuantizationType as Q  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402

import pixel_packed_shard as tool  # noqa: E402

N_LAYER, N_EMBD, N_FF = 4, 64, 256
PARENT = "sha256:" + "ab" * 32


def write(path: Path, tensors: dict[str, tuple[np.ndarray, Q]]) -> None:
    writer = gguf.GGUFWriter(str(path), "qwen3")
    writer.add_block_count(N_LAYER)
    writer.add_embedding_length(N_EMBD)
    writer.add_feed_forward_length(N_FF)
    for name, (data, qtype) in tensors.items():
        writer.add_tensor(name, data, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


class PackedShardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(7)
        origin, proxy = {}, {}
        for layer in range(N_LAYER):
            for name, shape in (("gate", (N_FF, N_EMBD)), ("up", (N_FF, N_EMBD)), ("down", (N_EMBD, N_FF))):
                values = rng.standard_normal(shape).astype(np.float32)
                packed = quantize(values, Q.Q4_0)
                origin[f"blk.{layer}.ffn_{name}.weight"] = (packed, Q.Q4_0)
                proxy[f"blk.{layer}.ffn_{name}.weight"] = (dequantize(packed, Q.Q4_0).astype(np.float16), Q.F16)
            origin[f"blk.{layer}.attn_q.weight"] = (rng.standard_normal((N_EMBD, N_EMBD)).astype(np.float32), Q.F32)
        self.origin, self.proxy = self.tmp / "origin.gguf", self.tmp / "proxy.gguf"
        write(self.origin, origin)
        write(self.proxy, proxy)
        self.origin_tensors = origin
        self.proxy_tensors = proxy

    def test_shard_copies_bytes_and_carries_layer_metadata(self):
        out = self.tmp / "shard" / "P.ffn.gguf"
        manifest = tool.build(self.origin, self.proxy, (1, 2), out, parent_sha256=PARENT, allowed_types=("Q4_0",))
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(manifest["layer_mask"], f"{0b110:016x}")
        self.assertTrue(all(row["exact_dequantization"]["mismatches"] == 0 for row in manifest["tensors"]))
        reader = gguf.GGUFReader(str(out))
        self.assertEqual({t.name for t in reader.tensors},
                         {f"blk.{layer}.ffn_{op}.weight" for layer in (1, 2) for op in ("gate", "up", "down")})
        for tensor in reader.tensors:
            self.assertEqual(np.asarray(tensor.data).tobytes(), self.origin_tensors[tensor.name][0].tobytes())
        meta = {k[len(tool.METADATA_PREFIX):]: f.contents() for k, f in reader.fields.items()
                if k.startswith(tool.METADATA_PREFIX)}
        self.assertEqual(meta["layers"], [1, 2])
        self.assertEqual(meta["parent_sha256"], PARENT)
        self.assertFalse(any(k.startswith("s42.ffn_shard.") for k in reader.fields))    # worker stays non-shard
        record = json.loads(out.with_name(out.name + ".index-record.json").read_text())
        self.assertEqual((record["layer_mask"], record["weight_type"], record["columns"]),
                         (f"{0b110:016x}", "Q4_0", N_FF))
        sys.path.insert(0, str(REPO_ROOT))
        from research_dev.scheduler.adapters.ffn_shards import FfnShardIndex
        index = FfnShardIndex.load(out.with_name("PIXEL_FFN_SHARDS.json"), "/data/local/tmp/x")
        self.assertEqual(index.records[0].layer_mask, 0b110)
        self.assertEqual(index.records[0].shard_sha256, manifest["shard_sha256"])

    def test_legacy_layout_has_no_extra_keys_and_is_deterministic(self):
        first = self.tmp / "a" / "P.ffn.gguf"
        second = self.tmp / "b" / "P.ffn.gguf"
        tool.build(self.origin, self.proxy, (0, 3), first, parent_sha256=PARENT, legacy=True, verify="sampled")
        tool.build(self.origin, self.proxy, (0, 3), second, parent_sha256=PARENT, legacy=True, verify="sampled")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        reader = gguf.GGUFReader(str(first))
        self.assertFalse(any(k.startswith(("s42.", "s43.")) for k in reader.fields))
        self.assertEqual(reader.fields["general.name"].contents(), tool.LEGACY_NAME)

    def test_refusals(self):
        with self.assertRaises(tool.PackedShardError):          # wrong allowed types
            tool.build(self.origin, self.proxy, (1,), self.tmp / "x" / "a.gguf", parent_sha256=PARENT,
                       allowed_types=("Q4_K", "Q6_K"))
        with self.assertRaises(tool.PackedShardError):          # beyond block_count
            tool.build(self.origin, self.proxy, (4,), self.tmp / "x" / "b.gguf", parent_sha256=PARENT)
        broken = dict(self.proxy_tensors)
        name = "blk.1.ffn_up.weight"
        values = broken[name][0].copy()
        values.view(np.uint16)[3, 5] ^= 1                       # one f16 bit off
        broken[name] = (values, Q.F16)
        bad_proxy = self.tmp / "bad.gguf"
        write(bad_proxy, broken)
        with self.assertRaisesRegex(tool.PackedShardError, "differ from the f16 proxy"):
            tool.build(self.origin, bad_proxy, (1,), self.tmp / "x" / "c.gguf", parent_sha256=PARENT)
        manifest = json.loads((self.tmp / "x" / "c.gguf.json").read_text())
        self.assertEqual(manifest["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
