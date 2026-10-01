#!/usr/bin/env python3
"""Pixel packed FFN shards for any layer set (WS11): the quantized origin's own FFN bytes, no requantization.

The desktop runs an f16 *proxy* that is the exact dequantization of a quantized origin file (Qwen3-14B:
Q4_K_M -> F16, Gemma-4-12B: Q4_0 -> F16). The Pixel's packed CPU worker computes on the origin's packed
blocks instead (4.5 bits/weight, 3.4x fewer bytes to stream). This tool generalizes the qualified builder
``reports/20260922-fast-path-M3/prepare_pixel_packed_weights.py`` (hard-wired to Qwen layers 18-23):

* any layer spec, any origin whose FFN tensors are in ``--allowed-types`` (default: the origin's own types,
  which must be Q4_K/Q6_K for the qualified worker, or Q4_0 for the WS11 worker patch);
* every copied tensor is byte-identical to the origin (re-hashed after writing);
* the exact-dequantization check is EXHAUSTIVE by default: every value of every copied tensor, dequantized
  and rounded to f16, must equal the proxy's f16 bits (``--verify sampled`` keeps the old 32-row check);
* ``--legacy`` writes the qualified file's exact layout (same name, keys, tensor order): layers 18-23 of the
  official Qwen3-14B Q4_K_M then reproduce ``QWEN_PACKED.ffn.gguf`` byte-for-byte (sha256 940f5f1f...);
  without ``--legacy`` the shard also carries ``s43.packed_shard.*`` metadata (inert for the worker, which
  enters shard mode only on ``s42.ffn_shard.version``): parent and origin sha256, layers, mask, geometry.

Outputs: ``<out>`` (the shard), ``<out>.json`` (manifest: per-tensor payload sha256, verification counts),
``<out>.index-record.json`` (its ``s42-ffn-shard-index-v1`` record) and ``PIXEL_FFN_SHARDS.json`` next to it (the
single-record index ``helper_phone_ffn_shards.<device>.index_path`` points at; it passes ``FfnShardIndex``).

    pixel_packed_shard.py ORIGIN.gguf PROXY_F16.gguf --layers 18-24 --out DIR/QWEN_PACKED_18_24.ffn.gguf \\
        --parent-sha256 sha256:d89e9e82... [--origin-sha256 sha256:500a8806...] [--session PIXEL10PRO0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "gguf-py") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "gguf-py"))

import gguf  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from ffn_shard_gguf import layer_mask, layer_spec, parse_layer_spec, sha256_file  # noqa: E402

METADATA_PREFIX = "s43.packed_shard."
MANIFEST_SCHEMA = "s43-pixel-packed-shard-manifest-v1"
FFN_TENSORS = ("ffn_gate", "ffn_up", "ffn_down")
LEGACY_NAME = "Private Pixel original Q4_K/Q6_K FFN slice"


class PackedShardError(ValueError):
    pass


def _field(reader: gguf.GGUFReader, key: str):
    field = reader.fields.get(key)
    return None if field is None else field.contents()


def _geometry(reader: gguf.GGUFReader) -> tuple[str, int, int, int]:
    architecture = _field(reader, "general.architecture")
    if not isinstance(architecture, str):
        raise PackedShardError("origin has no general.architecture")
    block_count = _field(reader, f"{architecture}.block_count")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    gate = tensors.get("blk.0.ffn_gate.weight")
    if not isinstance(block_count, int) or gate is None:
        raise PackedShardError("origin lacks block_count or blk.0.ffn_gate.weight")
    return architecture, block_count, int(gate.shape[0]), int(gate.shape[1])


def verify_exact(tensor, target, *, mode: str) -> dict[str, object]:
    """Dequantize ``tensor`` (all rows, or 32 sampled rows) and compare its f16 bits with ``target``."""
    if target.tensor_type != gguf.GGMLQuantizationType.F16 or not np.array_equal(tensor.shape, target.shape):
        raise PackedShardError(f"proxy geometry or type differs: {tensor.name}")
    rows = tensor.data.shape[0]
    if mode == "sampled":
        indices = np.unique(np.linspace(0, rows - 1, 32, dtype=int))
        decoded = gguf.quants.dequantize(tensor.data[indices], tensor.tensor_type).astype(np.float16)
        mismatches = int(np.count_nonzero(decoded.view(np.uint16) != target.data[indices].view(np.uint16)))
        checked = int(decoded.size)
    else:
        mismatches = checked = 0
        for start in range(0, rows, 256):
            decoded = gguf.quants.dequantize(tensor.data[start:start + 256], tensor.tensor_type).astype(np.float16)
            expected = target.data[start:start + 256].view(np.uint16)
            mismatches += int(np.count_nonzero(decoded.view(np.uint16) != expected))
            checked += int(decoded.size)
    return {"mode": mode, "values_checked": checked, "mismatches": mismatches}


def build(origin: Path, proxy: Path | None, layers: Sequence[int], output: Path, *, parent_sha256: str | None,
          origin_sha256: str | None = None, allowed_types: Sequence[str] = (), legacy: bool = False,
          verify: str = "exhaustive", session_id: str = "PIXEL10PRO0") -> dict[str, object]:
    if output.exists():
        raise PackedShardError(f"{output} already exists")
    if verify not in ("exhaustive", "sampled", "none"):
        raise PackedShardError("verify must be exhaustive, sampled or none")
    if verify != "none" and proxy is None:
        raise PackedShardError("the exact-dequantization check needs the f16 proxy")
    reader = gguf.GGUFReader(str(origin))
    architecture, block_count, n_embd, n_ff = _geometry(reader)
    if max(layers) >= block_count:
        raise PackedShardError(f"layer {max(layers)} >= block_count {block_count}")
    selected = {f"blk.{layer}.{name}.weight" for layer in layers for name in FFN_TENSORS}
    proxy_tensors = {} if proxy is None else {tensor.name: tensor for tensor in gguf.GGUFReader(str(proxy)).tensors}
    if proxy is not None:
        proxy_architecture, proxy_blocks, proxy_embd, proxy_ff = _geometry(gguf.GGUFReader(str(proxy)))
        if (proxy_architecture, proxy_blocks, proxy_embd, proxy_ff) != (architecture, block_count, n_embd, n_ff):
            raise PackedShardError("origin and proxy geometry differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output), architecture)
    writer.add_name(LEGACY_NAME if legacy else f"Pixel packed FFN slice {layer_spec(layers)}")
    writer.add_block_count(block_count)
    writer.add_embedding_length(n_embd)
    writer.add_feed_forward_length(n_ff)
    rows = []
    for tensor in reader.tensors:           # origin order, as the qualified builder did
        if tensor.name not in selected:
            continue
        type_name = gguf.GGMLQuantizationType(tensor.tensor_type).name
        if allowed_types and type_name not in allowed_types:
            raise PackedShardError(f"{tensor.name} is {type_name}, not one of {list(allowed_types)}")
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        row = {"name": tensor.name, "type": type_name, "shape": [int(value) for value in tensor.shape],
               "bytes": int(tensor.n_bytes), "payload_sha256": hashlib.sha256(tensor.data.tobytes()).hexdigest()}
        if verify != "none":
            row["exact_dequantization"] = verify_exact(tensor, proxy_tensors[tensor.name], mode=verify)
        rows.append(row)
    if len(rows) != 3 * len(layers):
        raise PackedShardError(f"found {len(rows)} of {3 * len(layers)} FFN tensors")
    types = sorted({row["type"] for row in rows})
    if not legacy:
        writer.add_uint32(METADATA_PREFIX + "version", 1)
        if parent_sha256:
            writer.add_string(METADATA_PREFIX + "parent_sha256", parent_sha256)
        if origin_sha256:
            writer.add_string(METADATA_PREFIX + "origin_sha256", origin_sha256)
        writer.add_uint64(METADATA_PREFIX + "n_embd", n_embd)
        writer.add_uint64(METADATA_PREFIX + "n_ff", n_ff)
        writer.add_uint64(METADATA_PREFIX + "layer_mask", layer_mask(layers))
        writer.add_array(METADATA_PREFIX + "layers", list(layers))
        writer.add_string(METADATA_PREFIX + "weight_types", ",".join(types))
        writer.add_string(METADATA_PREFIX + "session_id", session_id)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    copied = {tensor.name: tensor for tensor in gguf.GGUFReader(str(output)).tensors}
    for row in rows:
        if hashlib.sha256(copied[row["name"]].data.tobytes()).hexdigest() != row["payload_sha256"]:
            raise PackedShardError(f"{row['name']} payload changed while writing")
    mismatches = sum(row.get("exact_dequantization", {}).get("mismatches", 0) for row in rows)
    shard_sha256 = sha256_file(output)
    weight_type = types[0] if len(types) == 1 else "MIXED_" + "_".join(types)
    manifest = {
        "schema": MANIFEST_SCHEMA, "status": "PASS" if mismatches == 0 else "FAIL",
        "path": output.name, "shard_sha256": shard_sha256, "shard_bytes": output.stat().st_size,
        "origin": str(origin), "origin_sha256": origin_sha256, "proxy": None if proxy is None else str(proxy),
        "parent_sha256": parent_sha256, "architecture": architecture, "n_embd": n_embd, "n_ff": n_ff,
        "layers": list(layers), "layer_spec": layer_spec(layers), "layer_mask": f"{layer_mask(layers):016x}",
        "weight_types": types, "legacy_layout": legacy, "verify": verify, "tensors": rows,
        "weight_copy": "all tensor bytes exact (re-hashed after writing)",
        "scope": "stored weights only; execution arithmetic is qualified separately (numerical rows + server tokens)",
    }
    output.with_name(output.name + ".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if parent_sha256:
        record = {"path": output.name, "parent_sha256": parent_sha256, "shard_sha256": shard_sha256,
                  "layer_mask": f"{layer_mask(layers):016x}", "columns": n_ff, "n_ff": n_ff,
                  "shard_bytes": output.stat().st_size, "weight_type": weight_type, "session_id": session_id}
        output.with_name(output.name + ".index-record.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n")
        # the single-record helper index (helper_phone_ffn_shards.<device>.index_path) the scheduler loads
        index = {"schema": "s42-ffn-shard-index-v1", "parent_sha256": parent_sha256, "shards": [record]}
        output.with_name("PIXEL_FFN_SHARDS.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    if mismatches:
        raise PackedShardError(f"{mismatches} dequantized values differ from the f16 proxy")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("origin", type=Path, help="quantized origin GGUF (Q4_K_M / Q4_0)")
    parser.add_argument("proxy", type=Path, nargs="?", help="f16 proxy GGUF the desktop runs")
    parser.add_argument("--layers", required=True, help="e.g. 18-24")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-sha256", help="sha256:<hex> of the f16 proxy (the scheduler's artifact id)")
    parser.add_argument("--origin-sha256")
    parser.add_argument("--allowed-types", default="", help="comma list, e.g. Q4_K,Q6_K or Q4_0")
    parser.add_argument("--legacy", action="store_true", help="the qualified QWEN_PACKED layout, no metadata")
    parser.add_argument("--verify", default="exhaustive", choices=("exhaustive", "sampled", "none"))
    parser.add_argument("--session", default="PIXEL10PRO0")
    args = parser.parse_args(argv)
    try:
        manifest = build(args.origin, args.proxy, parse_layer_spec(args.layers), args.out,
                         parent_sha256=args.parent_sha256, origin_sha256=args.origin_sha256,
                         allowed_types=tuple(filter(None, args.allowed_types.split(","))), legacy=args.legacy,
                         verify=args.verify, session_id=args.session)
    except (PackedShardError, OSError, ValueError) as error:
        print(f"pixel_packed_shard: {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: manifest[key] for key in ("status", "path", "shard_sha256", "shard_bytes", "layer_spec",
                                                     "weight_types")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
