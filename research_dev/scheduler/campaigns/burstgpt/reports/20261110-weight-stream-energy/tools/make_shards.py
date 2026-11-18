#!/usr/bin/env python3
"""Write one f16 FFN shard (s42.ffn_shard format) of the first N layers of a model GGUF, for the
weight-stream energy benchmark. F16 parents are byte-copied; BF16 / F32 parents are converted to F16.

    make_shards.py MODEL.gguf OUT_DIR [--layers 0-7] [--parent-sha256 sha256:HEX]

A parent whose architecture the FFN worker does not accept by name (qwen2: same bias-free SwiGLU FFN as
qwen3) is written with the shard architecture "qwen3"; the manifest records the parent's own architecture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import os  # noqa: E402

REPO = Path(os.environ.get("STREAM_REPO") or Path(__file__).resolve().parents[7])
sys.path.insert(0, str(REPO / "gguf-py"))
sys.path.insert(0, str(REPO / "research_dev" / "scheduler" / "native"))
import gguf  # noqa: E402
import ffn_shard_gguf as S  # noqa: E402
from gguf.constants import GGMLQuantizationType as Q  # noqa: E402

WORKER_ARCH = {"qwen2": "qwen3"}


def as_f16(tensor) -> tuple[np.ndarray, str]:
    """(ne1, ne0) float16 array of a 2-D F16/BF16/F32 tensor."""
    kind = Q(tensor.tensor_type)
    ne0, ne1 = int(tensor.shape[0]), int(tensor.shape[1])
    raw = np.ascontiguousarray(tensor.data)
    if kind == Q.F16:
        return raw.view(np.float16).reshape(ne1, ne0), "copied"
    if kind == Q.BF16:
        bits = raw.view(np.uint16).reshape(ne1, ne0).astype(np.uint32) << 16
        return bits.view(np.float32).astype(np.float16), "bf16->f16"
    if kind == Q.F32:
        return raw.view(np.float32).reshape(ne1, ne0).astype(np.float16), "f32->f16"
    raise SystemExit(f"{tensor.name}: {kind.name} is not a float type")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--layers", default="0-7")
    ap.add_argument("--parent-sha256", default=None)
    args = ap.parse_args()
    layers = S.parse_layer_spec(args.layers)
    reader = gguf.GGUFReader(str(args.model))
    geo = S.parent_geometry(reader)
    parent = args.parent_sha256 or "sha256:" + S.sha256_file(args.model).removeprefix("sha256:")
    arch = WORKER_ARCH.get(geo.architecture, geo.architecture)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "HTP0.ffn.gguf"
    tensors = {t.name: t for t in reader.tensors}
    w = gguf.GGUFWriter(str(out), arch)
    w.add_name(f"{geo.name} ffn-shard f16 layers {args.layers}")
    w.add_block_count(geo.n_layer)
    w.add_embedding_length(geo.n_embd)
    w.add_feed_forward_length(geo.n_ff)
    P = S.METADATA_PREFIX
    w.add_uint32(P + "version", S.SHARD_FORMAT_VERSION)
    w.add_string(P + "parent_sha256", parent)
    w.add_uint64(P + "n_embd", geo.n_embd)
    w.add_uint64(P + "n_ff", geo.n_ff)
    w.add_uint64(P + "column_offset", 0)
    w.add_uint64(P + "columns", geo.n_ff)
    w.add_uint64(P + "layer_mask", S.layer_mask(layers))
    w.add_array(P + "layers", list(layers))
    w.add_string(P + "weight_type", "F16")
    w.add_string(P + "session_id", "HTP0")
    rows, how = [], set()
    for layer in layers:
        for suffix in S.FFN_TENSORS:
            name = f"blk.{layer}.{suffix}.weight"
            t = tensors[name]
            ne0, ne1 = int(t.shape[0]), int(t.shape[1])
            want = (geo.n_ff, geo.n_embd) if suffix == "ffn_down" else (geo.n_embd, geo.n_ff)
            if (ne0, ne1) != want:
                raise SystemExit(f"{name}: shape {ne0}x{ne1}, expected {want}")
            data, mode = as_f16(t)
            how.add(mode)
            w.add_tensor(name, data.view(np.uint8).reshape(ne1, ne0 * 2), raw_dtype=Q.F16)
            rows.append({"name": name, "ne": [ne0, ne1], "bytes": int(data.nbytes)})
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    manifest = {
        "model": str(args.model), "parent_sha256": parent, "parent_architecture": geo.architecture,
        "shard_architecture": arch, "parent_weight_type": geo.weight_type.name, "conversion": sorted(how),
        "n_layer": geo.n_layer, "n_embd": geo.n_embd, "n_ff": geo.n_ff, "layers": list(layers),
        "layer_bytes_f16": 3 * geo.n_embd * geo.n_ff * 2, "shard_bytes": out.stat().st_size,
        "shard_sha256": "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest() if out.stat().st_size < 2 << 30
        else S.sha256_file(out), "tensors": rows,
    }
    (args.out_dir / "HTP0.ffn.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "tensors"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
