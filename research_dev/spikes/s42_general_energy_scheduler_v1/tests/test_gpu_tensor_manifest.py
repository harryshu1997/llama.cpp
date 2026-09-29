#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
for path in (REPO_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_residency_v1.export_gpu_tensor_manifest import (  # noqa: E402
    ManifestError,
    canonical,
    gpu_geometry,
    hash_range,
    placement_record,
    select_tensors,
    select_stage_tensors,
    stage_placement_record,
)


def tensors(*, tied_output: bool) -> list[dict[str, object]]:
    names = [
        "token_embd.weight",
        "output_norm.weight",
        *(f"blk.{layer}.weight" for layer in range(4)),
    ]
    if not tied_output:
        names.append("output.weight")
    return [
        {
            "n_bytes": index + 1,
            "name": name,
            "raw_sha256": hashlib.sha256(name.encode("ascii")).hexdigest(),
        }
        for index, name in enumerate(names)
    ]


class GpuTensorManifestTests(unittest.TestCase):
    def test_llama_tail_layer_geometry_includes_output(self) -> None:
        self.assertEqual(gpu_geometry(4, 0), {
            "gpu_start_layer": 5,
            "output_layer_on_gpu": False,
            "repeating_layer_ids": (),
        })
        self.assertEqual(gpu_geometry(4, 1), {
            "gpu_start_layer": 4,
            "output_layer_on_gpu": True,
            "repeating_layer_ids": (),
        })
        self.assertEqual(gpu_geometry(4, 2), {
            "gpu_start_layer": 3,
            "output_layer_on_gpu": True,
            "repeating_layer_ids": (3,),
        })
        self.assertEqual(gpu_geometry(4, 5), {
            "gpu_start_layer": 0,
            "output_layer_on_gpu": True,
            "repeating_layer_ids": (0, 1, 2, 3),
        })

    def test_explicit_output_does_not_duplicate_embedding(self) -> None:
        selected = select_tensors(
            tensors(tied_output=False), block_count=4, n_gpu_layers=2
        )
        roles = {row["name"]: row["materialization_role"] for row in selected}
        self.assertEqual(roles, {
            "blk.3.weight": "REPEATING_LAYER",
            "output.weight": "OUTPUT_LAYER",
            "output_norm.weight": "OUTPUT_LAYER",
        })

    def test_tied_output_materializes_embedding_on_gpu(self) -> None:
        selected = select_tensors(
            tensors(tied_output=True), block_count=4, n_gpu_layers=1
        )
        roles = {row["name"]: row["materialization_role"] for row in selected}
        self.assertEqual(roles, {
            "output_norm.weight": "OUTPUT_LAYER",
            "token_embd.weight": "TIED_OUTPUT_DUPLICATE",
        })

    def test_zero_gpu_layers_has_empty_manifest(self) -> None:
        self.assertEqual(
            select_tensors(
                tensors(tied_output=True), block_count=4, n_gpu_layers=0
            ),
            [],
        )

    def test_placement_hash_binds_tensor_hashes_and_geometry(self) -> None:
        rows = tensors(tied_output=False)
        placement = placement_record(
            model_sha256="a" * 64,
            block_count=4,
            n_gpu_layers=2,
            tensors=rows,
        )
        claimed = placement.pop("placement_weight_sha256")
        self.assertEqual(
            claimed, hashlib.sha256(canonical(placement)).hexdigest()
        )
        self.assertEqual(placement["repeating_layer_ids"], [3])
        self.assertEqual(placement["selected_tensor_count"], 3)
        self.assertEqual(
            placement["selected_materialized_raw_bytes"],
            sum(row["n_bytes"] for row in placement["entries"]),
        )

    def test_nonterminal_stage_selects_only_repeating_slice(self) -> None:
        rows = tensors(tied_output=True)
        selected = select_stage_tensors(
            rows,
            block_count=4,
            layer_start=0,
            layer_end=1,
        )
        self.assertEqual(selected, [{
            "layer_id": 0,
            "materialization_role": "REPEATING_LAYER",
            "name": "blk.0.weight",
        }])

    def test_stage_hash_binds_raw_tensors_and_range(self) -> None:
        placement = stage_placement_record(
            model_sha256="a" * 64,
            block_count=4,
            layer_start=1,
            layer_end=3,
            tensors=tensors(tied_output=False),
        )
        claimed = placement.pop("stage_weight_sha256")
        self.assertEqual(claimed, hashlib.sha256(canonical(placement)).hexdigest())
        self.assertEqual(
            [row["layer_id"] for row in placement["entries"]],
            [1, 2],
        )

    def test_stage_must_be_nonterminal(self) -> None:
        with self.assertRaisesRegex(ManifestError, "nonterminal"):
            select_stage_tensors(
                tensors(tied_output=True),
                block_count=4,
                layer_start=0,
                layer_end=4,
            )

    def test_incomplete_block_coverage_fails_closed(self) -> None:
        rows = tensors(tied_output=False)
        rows = [row for row in rows if row["name"] != "blk.2.weight"]
        with self.assertRaisesRegex(
            ManifestError, "repeating-layer coverage"
        ):
            select_tensors(rows, block_count=4, n_gpu_layers=2)

    def test_hash_range_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bytes.bin"
            path.write_bytes(b"0123456789")
            self.assertEqual(
                hash_range(path, 2, 5),
                hashlib.sha256(b"23456").hexdigest(),
            )
            with self.assertRaisesRegex(ManifestError, "truncated"):
                hash_range(path, 8, 4)


if __name__ == "__main__":
    unittest.main()
