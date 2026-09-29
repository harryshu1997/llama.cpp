#!/usr/bin/env python3
"""Export exact GGUF tensor manifests for llama.cpp tail-layer offload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


SCHEMA = "s42-llama-gpu-tensor-manifest-v1"
BLOCK_PATTERN = re.compile(r"^blk\.([0-9]+)\.")


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise ManifestError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def hash_range(path: Path, offset: int, size: int) -> str:
    require(offset >= 0 and size > 0, "tensor byte range")
    digest = hashlib.sha256()
    remaining = size
    try:
        with path.open("rb") as stream:
            stream.seek(offset)
            while remaining:
                block = stream.read(min(4 * 1024 * 1024, remaining))
                require(block != b"", "truncated tensor byte range")
                digest.update(block)
                remaining -= len(block)
    except OSError as exc:
        raise ManifestError(f"cannot hash tensor range: {exc}") from exc
    return digest.hexdigest()


def block_id(name: str) -> int | None:
    match = BLOCK_PATTERN.match(name)
    return None if match is None else int(match.group(1))


def is_output_tensor(name: str) -> bool:
    return name.startswith("output.") or name.startswith("output_")


def gpu_geometry(block_count: int, n_gpu_layers: int) -> dict[str, Any]:
    require(block_count > 0, "block count")
    require(0 <= n_gpu_layers <= block_count + 1, "GPU layer count")
    gpu_start = max(block_count + 1 - n_gpu_layers, 0)
    active = min(n_gpu_layers, block_count + 1)
    repeating = tuple(
        layer
        for layer in range(block_count)
        if gpu_start <= layer < gpu_start + active
    )
    return {
        "gpu_start_layer": gpu_start,
        "output_layer_on_gpu": (
            gpu_start <= block_count < gpu_start + active
        ),
        "repeating_layer_ids": repeating,
    }


def select_tensors(
    tensors: Iterable[dict[str, Any]],
    *,
    block_count: int,
    n_gpu_layers: int,
) -> list[dict[str, Any]]:
    rows = tuple(tensors)
    names = [row.get("name") for row in rows]
    require(
        names
        and all(type(name) is str and name for name in names)
        and len(names) == len(set(names)),
        "tensor names",
    )
    observed_blocks = {
        layer for name in names if (layer := block_id(name)) is not None
    }
    require(
        observed_blocks == set(range(block_count)),
        "GGUF repeating-layer coverage",
    )
    geometry = gpu_geometry(block_count, n_gpu_layers)
    selected_layers = set(geometry["repeating_layer_ids"])
    has_output_weight = "output.weight" in names
    selected: list[dict[str, Any]] = []
    for row in rows:
        name = row["name"]
        layer = block_id(name)
        role = None
        if layer in selected_layers:
            role = "REPEATING_LAYER"
        elif geometry["output_layer_on_gpu"] and is_output_tensor(name):
            role = "OUTPUT_LAYER"
        elif (
            geometry["output_layer_on_gpu"]
            and not has_output_weight
            and name == "token_embd.weight"
        ):
            role = "TIED_OUTPUT_DUPLICATE"
        if role is not None:
            selected.append({
                "layer_id": layer,
                "materialization_role": role,
                "name": name,
            })
    if n_gpu_layers:
        require(selected, "nonempty GPU tensor selection")
    if geometry["output_layer_on_gpu"]:
        require(
            any(
                row["materialization_role"] in {
                    "OUTPUT_LAYER", "TIED_OUTPUT_DUPLICATE"
                }
                for row in selected
            ),
            "output-layer tensor selection",
        )
    return sorted(selected, key=lambda row: row["name"])


def select_stage_tensors(
    tensors: Iterable[dict[str, Any]],
    *,
    block_count: int,
    layer_start: int,
    layer_end: int,
) -> list[dict[str, Any]]:
    """Select CUDA weights for a nonterminal LayerSplit transformer stage."""
    require(block_count > 0, "block count")
    require(
        0 <= layer_start < layer_end < block_count,
        "nonterminal stage layer range",
    )
    rows = tuple(tensors)
    names = [row.get("name") for row in rows]
    require(
        names
        and all(type(name) is str and name for name in names)
        and len(names) == len(set(names)),
        "tensor names",
    )
    observed_blocks = {
        layer for name in names if (layer := block_id(name)) is not None
    }
    require(
        observed_blocks == set(range(block_count)),
        "GGUF repeating-layer coverage",
    )
    selected = [
        {
            "layer_id": layer,
            "materialization_role": "REPEATING_LAYER",
            "name": row["name"],
        }
        for row in rows
        if (
            (layer := block_id(row["name"])) is not None
            and layer_start <= layer < layer_end
        )
    ]
    require(bool(selected), "nonempty stage tensor selection")
    require(
        {row["layer_id"] for row in selected}
        == set(range(layer_start, layer_end)),
        "stage layer tensor coverage",
    )
    return sorted(selected, key=lambda row: row["name"])


def placement_record(
    *,
    model_sha256: str,
    block_count: int,
    n_gpu_layers: int,
    tensors: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = select_tensors(
        tensors,
        block_count=block_count,
        n_gpu_layers=n_gpu_layers,
    )
    by_name = {row["name"]: row for row in tensors}
    entries = [
        {
            **row,
            "n_bytes": by_name[row["name"]]["n_bytes"],
            "raw_sha256": by_name[row["name"]]["raw_sha256"],
        }
        for row in selected
    ]
    geometry = gpu_geometry(block_count, n_gpu_layers)
    unsigned = {
        "entries": entries,
        "model_sha256": model_sha256,
        "n_gpu_layers": n_gpu_layers,
        "output_layer_on_gpu": geometry["output_layer_on_gpu"],
        "repeating_layer_ids": list(geometry["repeating_layer_ids"]),
        "selected_materialized_raw_bytes": sum(
            row["n_bytes"] for row in entries
        ),
        "selected_tensor_count": len(entries),
    }
    return {
        **unsigned,
        "placement_weight_sha256": hashlib.sha256(
            canonical(unsigned)
        ).hexdigest(),
    }


def stage_placement_record(
    *,
    model_sha256: str,
    block_count: int,
    layer_start: int,
    layer_end: int,
    tensors: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = select_stage_tensors(
        tensors,
        block_count=block_count,
        layer_start=layer_start,
        layer_end=layer_end,
    )
    by_name = {row["name"]: row for row in tensors}
    entries = [
        {
            **row,
            "n_bytes": by_name[row["name"]]["n_bytes"],
            "raw_sha256": by_name[row["name"]]["raw_sha256"],
        }
        for row in selected
    ]
    require(
        all(
            type(row["raw_sha256"]) is str
            and len(row["raw_sha256"]) == 64
            for row in entries
        ),
        "stage tensor hashes",
    )
    unsigned = {
        "entries": entries,
        "layer_end": layer_end,
        "layer_start": layer_start,
        "model_sha256": model_sha256,
        "selected_materialized_raw_bytes": sum(
            row["n_bytes"] for row in entries
        ),
        "selected_tensor_count": len(entries),
    }
    return {
        **unsigned,
        "stage_weight_sha256": hashlib.sha256(canonical(unsigned)).hexdigest(),
    }


def export_manifest(
    *,
    model_path: Path,
    model_id: str,
    expected_model_sha256: str,
    expected_model_size: int,
    block_count: int,
    layer_counts: tuple[int, ...],
    gguf_python_path: Path,
) -> dict[str, Any]:
    require(model_path.is_absolute() and model_path.is_file(), "model path")
    require(
        model_path.stat().st_size == expected_model_size,
        "model size mismatch",
    )
    require(
        len(expected_model_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in expected_model_sha256
        ),
        "expected model SHA-256",
    )
    require(
        layer_counts
        and tuple(sorted(set(layer_counts))) == layer_counts,
        "ordered unique GPU layer counts",
    )
    require(gguf_python_path.is_dir(), "GGUF Python path")
    sys.path.insert(0, str(gguf_python_path))
    try:
        from gguf import GGUFReader  # type: ignore
    except (ImportError, OSError) as exc:
        raise ManifestError(f"cannot import GGUFReader: {exc}") from exc

    model_digest = sha256(model_path)
    require(model_digest == expected_model_sha256, "model SHA-256 mismatch")
    reader = GGUFReader(model_path, "r")
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
    require(
        descriptors
        and len({row["name"] for row in descriptors}) == len(descriptors),
        "GGUF tensor descriptors",
    )
    file_size = model_path.stat().st_size
    for row in descriptors:
        require(
            row["data_offset"] >= int(reader.data_offset)
            and row["n_bytes"] > 0
            and row["data_offset"] + row["n_bytes"] <= file_size,
            f"GGUF tensor byte range: {row['name']}",
        )
    selected_names: set[str] = set()
    descriptor_stub = [{"name": row["name"]} for row in descriptors]
    for layers in layer_counts:
        selected_names.update(
            row["name"]
            for row in select_tensors(
                descriptor_stub,
                block_count=block_count,
                n_gpu_layers=layers,
            )
        )
    hashed_tensors = []
    for row in descriptors:
        if row["name"] not in selected_names:
            continue
        hashed_tensors.append({
            **row,
            "raw_sha256": hash_range(
                model_path, row["data_offset"], row["n_bytes"]
            ),
        })
    require(
        {row["name"] for row in hashed_tensors} == selected_names,
        "selected tensor hash coverage",
    )
    hashes_by_name = {
        row["name"]: row["raw_sha256"] for row in hashed_tensors
    }
    placement_tensors = [
        {
            **row,
            "raw_sha256": hashes_by_name.get(row["name"]),
        }
        for row in descriptors
    ]
    placements = [
        placement_record(
            model_sha256=model_digest,
            block_count=block_count,
            n_gpu_layers=layers,
            tensors=placement_tensors,
        )
        for layers in layer_counts
    ]
    output: dict[str, Any] = {
        "assignment_rule": {
            "input_layer": "CPU",
            "source": "src/llama-model.cpp:tail-layer-offload",
            "tensor_roles": "src/llama-model-loader.cpp:llm_tensor_layer",
        },
        "block_count": block_count,
        "exporter": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "gguf": {
            "alignment": int(reader.alignment),
            "data_offset": int(reader.data_offset),
            "tensor_count": len(descriptors),
            "tensor_layout_sha256": hashlib.sha256(
                canonical(descriptors)
            ).hexdigest(),
        },
        "model": {
            "id": model_id,
            "path": str(model_path),
            "sha256": model_digest,
            "size_bytes": file_size,
        },
        "placements": placements,
        "schema": SCHEMA,
        "selected_tensors": hashed_tensors,
        "status": "EXACT_GGUF_TENSOR_RANGES",
    }
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-model-size", type=int, required=True)
    parser.add_argument("--block-count", type=int, required=True)
    parser.add_argument(
        "--gpu-layers", type=int, action="append", required=True
    )
    parser.add_argument("--gguf-python-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an absolute new path")
    try:
        value = export_manifest(
            model_path=args.model,
            model_id=args.model_id,
            expected_model_sha256=args.expected_model_sha256,
            expected_model_size=args.expected_model_size,
            block_count=args.block_count,
            layer_counts=tuple(sorted(set(args.gpu_layers))),
            gguf_python_path=args.gguf_python_path,
        )
        args.output.write_bytes(canonical(value))
    except (ManifestError, OSError, ValueError) as exc:
        parser.exit(2, f"GPU tensor manifest export failed: {exc}\n")
    print(json.dumps({
        "model": value["model"]["id"],
        "placements": [
            {
                "n_gpu_layers": row["n_gpu_layers"],
                "selected_materialized_raw_bytes": row[
                    "selected_materialized_raw_bytes"
                ],
                "selected_tensor_count": row["selected_tensor_count"],
            }
            for row in value["placements"]
        ],
        "record_sha256": value["record_sha256"],
        "status": value["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
