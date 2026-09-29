#!/usr/bin/env python3
"""Export exact GGUF tensor hashes for a nonterminal CUDA layer stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from export_gpu_tensor_manifest import (
    ManifestError,
    canonical,
    hash_range,
    require,
    select_stage_tensors,
    sha256,
    stage_placement_record,
)


SCHEMA = "s42-llama-gpu-stage-tensor-manifest-v1"


def export_stage_manifest(
    *,
    model_path: Path,
    model_id: str,
    expected_model_sha256: str,
    expected_model_size: int,
    block_count: int,
    layer_start: int,
    layer_end: int,
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
        0 <= layer_start < layer_end < block_count,
        "nonterminal stage layer range",
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
    selected_names = {
        row["name"]
        for row in select_stage_tensors(
            ({"name": row["name"]} for row in descriptors),
            block_count=block_count,
            layer_start=layer_start,
            layer_end=layer_end,
        )
    }
    hashed_tensors = [
        {
            **row,
            "raw_sha256": hash_range(
                model_path, row["data_offset"], row["n_bytes"]
            ),
        }
        for row in descriptors
        if row["name"] in selected_names
    ]
    require(
        {row["name"] for row in hashed_tensors} == selected_names,
        "selected tensor hash coverage",
    )
    placement = stage_placement_record(
        model_sha256=model_digest,
        block_count=block_count,
        layer_start=layer_start,
        layer_end=layer_end,
        tensors=hashed_tensors + [
            {
                "n_bytes": row["n_bytes"],
                "name": row["name"],
                "raw_sha256": None,
            }
            for row in descriptors
            if row["name"] not in selected_names
        ],
    )
    output: dict[str, Any] = {
        "assignment_rule": {
            "input_layer": "CPU",
            "nonterminal_output_layer": "ABSENT",
            "repeating_stage_layers": "CUDA0",
            "source": "src/llama-model.cpp:load_tensors",
            "stage_source": "src/models/gemma4.cpp:load_arch_tensors",
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
        "schema": SCHEMA,
        "selected_tensors": hashed_tensors,
        "stage_placement": placement,
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
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--gguf-python-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an absolute new path")
    try:
        value = export_stage_manifest(
            model_path=args.model,
            model_id=args.model_id,
            expected_model_sha256=args.expected_model_sha256,
            expected_model_size=args.expected_model_size,
            block_count=args.block_count,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            gguf_python_path=args.gguf_python_path,
        )
        args.output.write_bytes(canonical(value))
    except (ManifestError, OSError, ValueError) as exc:
        parser.exit(2, f"GPU-stage tensor manifest export failed: {exc}\n")
    placement = value["stage_placement"]
    print(json.dumps({
        "layer_end": placement["layer_end"],
        "layer_start": placement["layer_start"],
        "record_sha256": value["record_sha256"],
        "selected_materialized_raw_bytes": placement[
            "selected_materialized_raw_bytes"
        ],
        "selected_tensor_count": placement["selected_tensor_count"],
        "stage_weight_sha256": placement["stage_weight_sha256"],
        "status": value["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
