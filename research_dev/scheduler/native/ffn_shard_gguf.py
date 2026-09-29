#!/usr/bin/env python3
"""Write per-session FFN shard GGUF files for the phone FFN split worker.

The worker (examples/layersplit/ffn-split-worker.cpp) serves, for each selected
layer, the last ``columns`` units of the FFN intermediate dimension: rows
``[n_ff - columns, n_ff)`` of ``ffn_gate``/``ffn_up`` and the matching columns
of ``ffn_down``.  Reading those slices out of the complete model GGUF means
reading every selected FFN matrix in full.  This tool materializes the slices
offline so a session opens a file that holds exactly what it loads:

    qwen/HTP0.ffn.gguf     layers 0-5,   suffix of 4096 columns
    qwen/HTP0.ffn.json     shard sha256, parent sha256, geometry
    qwen/FFN_SHARDS.json   index of every shard written by one invocation

Shard GGUF metadata (namespace ``s42.ffn_shard``):

    version        1
    parent_sha256  "sha256:<64 hex>" of the complete model GGUF
    n_embd, n_ff   parent FFN geometry
    column_offset  n_ff - columns (the stored slice is always the suffix)
    columns        stored suffix width (the maximum useful slice)
    layer_mask     uint64 bit mask of stored layers
    layers         uint32 array of stored layers
    weight_type    ggml type name of the FFN weights

The worker may serve any ``--columns`` up to the stored width and any layer
subset of the stored mask; it computes the same weight hash as it would from
the complete model, so scheduler identities do not change.

Usage:
    ffn_shard_gguf.py MODEL.gguf --parent-sha256 sha256:... --out-dir DIR \\
        --shard HTP0=0-5:4096 --shard HTP1=6-11:4096 [--verify-parent]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "gguf-py") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "gguf-py"))

import gguf  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType  # noqa: E402

SHARD_FORMAT_VERSION = 1
METADATA_PREFIX = "s42.ffn_shard."
INDEX_SCHEMA = "s42-ffn-shard-index-v1"
MANIFEST_SCHEMA = "s42-ffn-shard-manifest-v1"
FFN_TENSORS = ("ffn_gate", "ffn_up", "ffn_down")


class FfnShardError(ValueError):
    pass


def parse_layer_spec(text: str) -> tuple[int, ...]:
    """Same grammar as the worker: ``0-5``, ``7``, ``0-2,5``."""
    layers: set[int] = set()
    for item in text.split(","):
        if not item:
            raise FfnShardError(f"empty layer item in {text!r}")
        first, _, last = item.partition("-")
        try:
            begin = int(first)
            end = int(last) if last else begin
        except ValueError as error:
            raise FfnShardError(f"invalid layer item {item!r}") from error
        if begin < 0 or end < begin or end >= 64:
            raise FfnShardError(f"layer range {item!r} is out of 0..63")
        layers.update(range(begin, end + 1))
    if not layers:
        raise FfnShardError("layer spec selects no layers")
    return tuple(sorted(layers))


def layer_mask(layers: Sequence[int]) -> int:
    mask = 0
    for layer in layers:
        mask |= 1 << layer
    return mask


def layer_spec(layers: Sequence[int]) -> str:
    """Compact ``a-b,c`` spelling accepted by the worker."""
    parts = []
    ordered = sorted(layers)
    start = previous = ordered[0]
    for layer in ordered[1:]:
        if layer == previous + 1:
            previous = layer
            continue
        parts.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = layer
    parts.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(parts)


def row_bytes(quant_type: GGMLQuantizationType, elements: int) -> int:
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if elements % block_size != 0:
        raise FfnShardError(
            f"{elements} elements do not align to {quant_type.name} block {block_size}"
        )
    return elements // block_size * type_size


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_text(value: str) -> str:
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(c not in "0123456789abcdef" for c in value[7:])
    ):
        raise FfnShardError(f"invalid sha256 text {value!r}")
    return value


@dataclass(frozen=True)
class ShardRequest:
    session_id: str
    layers: tuple[int, ...]
    columns: int

    @classmethod
    def parse(cls, text: str) -> "ShardRequest":
        session_id, sep, rest = text.partition("=")
        spec, sep2, columns_text = rest.rpartition(":")
        if not sep or not sep2 or not session_id.isidentifier() and not all(
            c.isalnum() or c in "._-" for c in session_id
        ):
            raise FfnShardError(f"shard must be SESSION=LAYERS:COLUMNS, got {text!r}")
        try:
            columns = int(columns_text)
        except ValueError as error:
            raise FfnShardError(f"invalid column count in {text!r}") from error
        if columns <= 0:
            raise FfnShardError(f"column count must be positive in {text!r}")
        return cls(session_id, parse_layer_spec(spec), columns)


@dataclass(frozen=True)
class ParentGeometry:
    architecture: str
    name: str
    n_layer: int
    n_embd: int
    n_ff: int
    weight_type: GGMLQuantizationType


def _field_value(reader: gguf.GGUFReader, key: str):
    field = reader.fields.get(key)
    if field is None:
        return None
    return field.contents()


def parent_geometry(reader: gguf.GGUFReader) -> ParentGeometry:
    architecture = _field_value(reader, "general.architecture")
    if not isinstance(architecture, str):
        raise FfnShardError("parent GGUF has no general.architecture")
    name = _field_value(reader, "general.name")
    n_layer = _field_value(reader, f"{architecture}.block_count")
    if not isinstance(n_layer, int) or n_layer <= 0:
        raise FfnShardError("parent GGUF has no block_count")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    gate = tensors.get("blk.0.ffn_gate.weight")
    if gate is None:
        raise FfnShardError("parent GGUF has no blk.0.ffn_gate.weight")
    n_embd, n_ff = int(gate.shape[0]), int(gate.shape[1])
    return ParentGeometry(
        architecture=architecture,
        name=name if isinstance(name, str) else architecture,
        n_layer=n_layer,
        n_embd=n_embd,
        n_ff=n_ff,
        weight_type=GGMLQuantizationType(gate.tensor_type),
    )


def _tensor_bytes(tensor: gguf.ReaderTensor) -> np.ndarray:
    """Tensor payload as a (ne1, bytes_per_row) uint8 view."""
    ne0, ne1 = int(tensor.shape[0]), int(tensor.shape[1])
    per_row = row_bytes(GGMLQuantizationType(tensor.tensor_type), ne0)
    raw = np.ascontiguousarray(tensor.data).view(np.uint8).reshape(-1)
    if raw.size != per_row * ne1:
        raise FfnShardError(f"tensor {tensor.name} payload size is inconsistent")
    return raw.reshape(ne1, per_row)


def slice_suffix_rows(tensor: gguf.ReaderTensor, columns: int) -> np.ndarray:
    """Rows [ne1 - columns, ne1) of a (ne0=n_embd, ne1=n_ff) gate/up matrix."""
    data = _tensor_bytes(tensor)
    if columns > data.shape[0]:
        raise FfnShardError(f"{tensor.name} has only {data.shape[0]} rows")
    return np.ascontiguousarray(data[data.shape[0] - columns:, :])


def slice_suffix_columns(tensor: gguf.ReaderTensor, columns: int) -> np.ndarray:
    """Columns [ne0 - columns, ne0) of a (ne0=n_ff, ne1=n_embd) down matrix."""
    data = _tensor_bytes(tensor)
    quant_type = GGMLQuantizationType(tensor.tensor_type)
    ne0 = int(tensor.shape[0])
    if columns > ne0:
        raise FfnShardError(f"{tensor.name} has only {ne0} columns")
    begin = row_bytes(quant_type, ne0 - columns)
    end = row_bytes(quant_type, ne0)
    return np.ascontiguousarray(data[:, begin:end])


def write_shard(
    reader: gguf.GGUFReader,
    geometry: ParentGeometry,
    parent_sha256: str,
    request: ShardRequest,
    output: Path,
) -> dict:
    if request.columns > geometry.n_ff:
        raise FfnShardError(
            f"{request.session_id}: {request.columns} columns exceed n_ff {geometry.n_ff}"
        )
    block_size = GGML_QUANT_SIZES[geometry.weight_type][0]
    offset = geometry.n_ff - request.columns
    if request.columns % block_size != 0 or offset % block_size != 0:
        raise FfnShardError(
            f"{request.session_id}: suffix [{offset},{geometry.n_ff}) does not align "
            f"to {geometry.weight_type.name} block {block_size}"
        )
    if request.layers[-1] >= geometry.n_layer:
        raise FfnShardError(
            f"{request.session_id}: layer {request.layers[-1]} >= block_count {geometry.n_layer}"
        )
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    writer = gguf.GGUFWriter(str(output), geometry.architecture)
    writer.add_name(f"{geometry.name} ffn-shard {request.session_id}")
    writer.add_block_count(geometry.n_layer)
    writer.add_embedding_length(geometry.n_embd)
    writer.add_feed_forward_length(geometry.n_ff)
    writer.add_uint32(METADATA_PREFIX + "version", SHARD_FORMAT_VERSION)
    writer.add_string(METADATA_PREFIX + "parent_sha256", parent_sha256)
    writer.add_uint64(METADATA_PREFIX + "n_embd", geometry.n_embd)
    writer.add_uint64(METADATA_PREFIX + "n_ff", geometry.n_ff)
    writer.add_uint64(METADATA_PREFIX + "column_offset", offset)
    writer.add_uint64(METADATA_PREFIX + "columns", request.columns)
    writer.add_uint64(METADATA_PREFIX + "layer_mask", layer_mask(request.layers))
    writer.add_array(METADATA_PREFIX + "layers", list(request.layers))
    writer.add_string(METADATA_PREFIX + "weight_type", geometry.weight_type.name)
    writer.add_string(METADATA_PREFIX + "session_id", request.session_id)
    tensor_rows = []
    for layer in request.layers:
        prefix = f"blk.{layer}."
        for suffix in FFN_TENSORS:
            name = prefix + suffix + ".weight"
            tensor = tensors.get(name)
            if tensor is None:
                raise FfnShardError(f"parent GGUF lacks {name}")
            if GGMLQuantizationType(tensor.tensor_type) != geometry.weight_type:
                raise FfnShardError(f"{name} type differs from blk.0 FFN type")
            ne0, ne1 = int(tensor.shape[0]), int(tensor.shape[1])
            if suffix == "ffn_down":
                if (ne0, ne1) != (geometry.n_ff, geometry.n_embd):
                    raise FfnShardError(f"{name} shape {ne0}x{ne1} is not n_ff x n_embd")
                payload = slice_suffix_columns(tensor, request.columns)
                logical = (geometry.n_embd, request.columns)  # (ne1, ne0)
            else:
                if (ne0, ne1) != (geometry.n_embd, geometry.n_ff):
                    raise FfnShardError(f"{name} shape {ne0}x{ne1} is not n_embd x n_ff")
                payload = slice_suffix_rows(tensor, request.columns)
                logical = (request.columns, geometry.n_embd)  # (ne1, ne0)
            # payload is a (ne1, bytes_per_row) uint8 view; gguf-py derives the
            # logical shape from the byte shape and raw_dtype.
            writer.add_tensor(name, payload, raw_dtype=geometry.weight_type)
            tensor_rows.append({
                "name": name,
                "ne": [logical[1], logical[0]],
                "type": geometry.weight_type.name,
                "bytes": int(payload.nbytes),
            })
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "session_id": request.session_id,
        "path": output.name,
        "shard_sha256": sha256_file(output),
        "shard_bytes": output.stat().st_size,
        "parent_sha256": parent_sha256,
        "architecture": geometry.architecture,
        "n_embd": geometry.n_embd,
        "n_ff": geometry.n_ff,
        "column_offset": offset,
        "columns": request.columns,
        "layers": list(request.layers),
        "layer_spec": layer_spec(request.layers),
        "layer_mask": f"{layer_mask(request.layers):016x}",
        "weight_type": geometry.weight_type.name,
        "tensors": tensor_rows,
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return manifest


def write_shards(
    model: Path,
    parent_sha256: str,
    out_dir: Path,
    requests: Sequence[ShardRequest],
    *,
    verify_parent: bool = False,
) -> dict:
    parent_sha256 = _sha256_text(parent_sha256)
    if verify_parent:
        measured = sha256_file(model)
        if measured != parent_sha256:
            raise FfnShardError(
                f"parent sha256 {measured} differs from declared {parent_sha256}"
            )
    if len({request.session_id for request in requests}) != len(requests):
        raise FfnShardError("duplicate session ids")
    reader = gguf.GGUFReader(str(model))
    geometry = parent_geometry(reader)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for request in requests:
        output = out_dir / f"{request.session_id}.ffn.gguf"
        if output.exists():
            raise FfnShardError(f"{output} already exists")
        shards.append(write_shard(reader, geometry, parent_sha256, request, output))
    index = {
        "schema": INDEX_SCHEMA,
        "parent_model": model.name,
        "parent_sha256": parent_sha256,
        "parent_verified": bool(verify_parent),
        "architecture": geometry.architecture,
        "n_embd": geometry.n_embd,
        "n_ff": geometry.n_ff,
        "weight_type": geometry.weight_type.name,
        "shards": shards,
    }
    (out_dir / "FFN_SHARDS.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return index


def read_shard_metadata(path: Path) -> dict:
    """Decode the ``s42.ffn_shard`` keys of one shard GGUF."""
    reader = gguf.GGUFReader(str(path))
    result = {}
    for key, field in reader.fields.items():
        if key.startswith(METADATA_PREFIX):
            result[key[len(METADATA_PREFIX):]] = field.contents()
    result["tensors"] = {
        tensor.name: (int(tensor.shape[0]), int(tensor.shape[1]))
        for tensor in reader.tensors
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", type=Path, help="complete model GGUF")
    parser.add_argument("--parent-sha256", required=True,
                        help="sha256:<hex> of the complete model GGUF")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard", action="append", required=True,
                        metavar="SESSION=LAYERS:COLUMNS",
                        help="e.g. HTP0=0-5:4096 (repeatable)")
    parser.add_argument("--verify-parent", action="store_true",
                        help="hash the complete model and require it to match")
    args = parser.parse_args(argv)
    try:
        requests = [ShardRequest.parse(text) for text in args.shard]
        index = write_shards(
            args.model, args.parent_sha256, args.out_dir, requests,
            verify_parent=args.verify_parent,
        )
    except (FfnShardError, OSError) as error:
        print(f"ffn_shard_gguf: {error}", file=sys.stderr)
        return 1
    for shard in index["shards"]:
        print(
            f"{shard['session_id']}: layers {shard['layer_spec']} "
            f"columns {shard['columns']} -> {shard['path']} "
            f"({shard['shard_bytes']} bytes, {shard['shard_sha256']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
