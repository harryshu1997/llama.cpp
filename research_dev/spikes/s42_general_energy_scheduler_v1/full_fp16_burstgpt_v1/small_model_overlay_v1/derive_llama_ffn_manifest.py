#!/usr/bin/env python3
"""Derive an exact dense Llama FFN split contract from a GGUF artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


SCHEMA = "s42-llama-dense-ffn-manifest-v1"
MODEL_ID = "llama-3.2-1b-instruct-q4_0"
FFN_ROLES = ("ffn_gate", "ffn_up", "ffn_down")


class ManifestError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def range_sha256(path: Path, offset: int, size: int) -> str:
    require(offset >= 0 and size > 0, "tensor byte range")
    value = hashlib.sha256()
    remaining = size
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            require(bool(block), "truncated tensor byte range")
            value.update(block)
            remaining -= len(block)
    return value.hexdigest()


def dense_ffn_name(layer: int, role: str) -> str:
    require(layer >= 0 and role in FFN_ROLES, "dense FFN tensor identity")
    return f"blk.{layer}.{role}.weight"


def select_dense_ffn_tensors(
    descriptors: Iterable[dict[str, Any]],
    *,
    block_count: int,
    n_embd: int,
    n_ff: int,
) -> list[dict[str, Any]]:
    require(block_count > 0 and block_count <= 64, "block count")
    require(n_embd > 0 and n_ff > 0, "dense FFN geometry")
    rows = tuple(descriptors)
    by_name = {row.get("name"): row for row in rows}
    require(
        len(by_name) > 0
        and None not in by_name
        and len(by_name) == len(rows),
        "unique tensor descriptors",
    )
    selected = []
    for layer in range(block_count):
        for role in FFN_ROLES:
            name = dense_ffn_name(layer, role)
            require(name in by_name, f"missing dense FFN tensor: {name}")
            row = dict(by_name[name])
            expected = (
                [n_ff, n_embd]
                if role == "ffn_down"
                else [n_embd, n_ff]
            )
            require(row.get("shape") == expected, f"FFN tensor shape: {name}")
            require(
                type(row.get("n_bytes")) is int and row["n_bytes"] > 0,
                f"FFN tensor bytes: {name}",
            )
            row.update({"layer_id": layer, "role": role})
            selected.append(row)
    return selected


def aggregate_op(manifest: dict[str, Any], m: int) -> dict[str, Any]:
    require(m > 0 and m <= manifest["split_contract"]["max_tokens"], "M")
    geometry = manifest["geometry"]
    layers = len(geometry["resident_layer_ids"])
    n_embd = geometry["n_embd"]
    n_ff = geometry["n_ff"]
    io_bytes = layers * m * n_embd * 2
    return {
        "allowed_devices": ["cpu", "phone"],
        "compute_ops": layers * 6 * m * n_embd * n_ff,
        "input_bytes": io_bytes,
        "kernel_family": "fused_ffn_swiglu",
        "kind": "matmul",
        "layer_id": f"layers-0-{layers - 1}-dense-ffn",
        "op_id": f"llama-dense-ffn-m{m}",
        "output_bytes": io_bytes,
        "quantization": manifest["geometry"]["quantization"],
        "shape": {"k": n_embd, "m": m, "n": n_ff},
        "split_quantum_n": manifest["split_contract"]["column_quantum"],
        "weight_bytes": manifest["resident_slice"]["raw_bytes"],
        "weight_id": manifest["resident_slice"]["weight_sha256"],
    }


def derive(
    *,
    model_path: Path,
    model_id: str,
    expected_sha256: str,
    expected_size: int,
    gguf_python_path: Path,
    resident_layers: int,
) -> dict[str, Any]:
    require(model_path.is_absolute() and model_path.is_file(), "model path")
    require(model_path.stat().st_size == expected_size, "model size")
    require(
        len(expected_sha256) == 64
        and all(value in "0123456789abcdef" for value in expected_sha256),
        "expected model SHA-256",
    )
    require(gguf_python_path.is_dir(), "GGUF Python path")
    sys.path.insert(0, str(gguf_python_path))
    try:
        from gguf import GGUFReader  # type: ignore
    except (ImportError, OSError) as exc:
        raise ManifestError(f"cannot import GGUFReader: {exc}") from exc

    model_hash = file_sha256(model_path)
    require(model_hash == expected_sha256, "model SHA-256")
    reader = GGUFReader(model_path, "r")
    architecture = reader.fields["general.architecture"].contents()
    require(architecture == "llama", "GGUF architecture")

    def metadata_integer(suffix: str) -> int:
        field = reader.fields.get(f"{architecture}.{suffix}")
        require(field is not None, f"GGUF metadata: {suffix}")
        value = field.contents()
        require(type(value) is int and value > 0, f"GGUF integer: {suffix}")
        return value

    block_count = metadata_integer("block_count")
    require(0 < resident_layers <= block_count, "resident layer count")
    n_embd = metadata_integer("embedding_length")
    n_ff = metadata_integer("feed_forward_length")
    descriptors = [
        {
            "data_offset": int(tensor.data_offset),
            "n_bytes": int(tensor.n_bytes),
            "name": tensor.name,
            "shape": [int(value) for value in tensor.shape],
            "tensor_type": tensor.tensor_type.name,
        }
        for tensor in reader.tensors
    ]
    selected = select_dense_ffn_tensors(
        descriptors,
        block_count=resident_layers,
        n_embd=n_embd,
        n_ff=n_ff,
    )
    file_size = model_path.stat().st_size
    for row in selected:
        require(
            row["data_offset"] >= int(reader.data_offset)
            and row["data_offset"] + row["n_bytes"] <= file_size,
            f"tensor file range: {row['name']}",
        )
        row["raw_sha256"] = range_sha256(
            model_path, row["data_offset"], row["n_bytes"]
        )
    quantizations = {row["tensor_type"] for row in selected}
    require(len(quantizations) == 1, "uniform dense FFN quantization")
    tensor_type = next(iter(quantizations))
    require(tensor_type == "Q4_0", "qualified Llama FFN quantization")
    quantization = tensor_type.lower()
    slice_identity = [
        {
            "layer_id": row["layer_id"],
            "n_bytes": row["n_bytes"],
            "name": row["name"],
            "raw_sha256": row["raw_sha256"],
            "role": row["role"],
            "shape": row["shape"],
            "tensor_type": row["tensor_type"],
        }
        for row in selected
    ]
    result: dict[str, Any] = {
        "geometry": {
            "activation": "swiglu",
            "block_count": block_count,
            "resident_layer_ids": list(range(resident_layers)),
            "n_embd": n_embd,
            "n_ff": n_ff,
            "quantization": quantization,
        },
        "gguf": {
            "alignment": int(reader.alignment),
            "data_offset": int(reader.data_offset),
            "tensor_count": len(descriptors),
        },
        "model": {
            "id": model_id,
            "path": str(model_path),
            "sha256": model_hash,
            "size_bytes": file_size,
        },
        "resident_slice": {
            "raw_bytes": sum(row["n_bytes"] for row in selected),
            "tensor_count": len(selected),
            "tensors": slice_identity,
            "weight_sha256": hashlib.sha256(canonical(slice_identity)).hexdigest(),
        },
        "schema": SCHEMA,
        "split_contract": {
            "column_quantum": 1024,
            "f16_io": True,
            "layer_mask": (1 << resident_layers) - 1,
            "max_columns": n_ff,
            "max_tokens": 512,
            "phone_backend": "HTP3",
            "transport": "ncm",
        },
        "status": "EXACT_GGUF_FFN_SLICE",
    }
    result["aggregate_ops"] = [
        aggregate_op(result, m) for m in (1, 8, 32, 128, 512)
    ]
    result["record_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--gguf-python-path", type=Path, required=True)
    parser.add_argument("--resident-layers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists(), "new output")
    value = derive(
        model_path=args.model,
        model_id=args.model_id,
        expected_sha256=args.expected_sha256,
        expected_size=args.expected_size,
        gguf_python_path=args.gguf_python_path,
        resident_layers=args.resident_layers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(value))
    print(json.dumps({
        "output": str(args.output),
        "raw_bytes": value["resident_slice"]["raw_bytes"],
        "record_sha256": value["record_sha256"],
        "status": value["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
