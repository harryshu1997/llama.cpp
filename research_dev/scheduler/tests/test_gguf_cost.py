#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from research_dev.scheduler import GGUFModelManifestLoader
from research_dev.scheduler._internal.model_manifest import _field_int


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGMLQuantizationType, GGUFWriter, quantize


def write_synthetic_gguf(
    path: Path,
    *,
    block_count: int = 1,
    head_count_kv_by_layer: tuple[int, ...] | None = None,
    sliding_window: int | None = None,
    sliding_window_pattern: tuple[bool, ...] | None = None,
    sliding_window_key_length: int | None = None,
    sliding_window_value_length: int | None = None,
    explicit_kv_geometry: bool = True,
) -> dict[str, int]:
    writer = GGUFWriter(path, "synthetic_transformer")
    writer.add_context_length(128)
    writer.add_embedding_length(32)
    writer.add_block_count(block_count)
    writer.add_feed_forward_length(128)
    writer.add_head_count(4)
    if explicit_kv_geometry:
        writer.add_head_count_kv(
            2 if head_count_kv_by_layer is None
            else head_count_kv_by_layer
        )
        writer.add_key_length(8)
        writer.add_value_length(8)
    if sliding_window is not None:
        writer.add_sliding_window(sliding_window)
    if sliding_window_pattern is not None:
        writer.add_sliding_window_pattern(sliding_window_pattern)
    if sliding_window_key_length is not None:
        writer.add_key_length_swa(sliding_window_key_length)
    if sliding_window_value_length is not None:
        writer.add_value_length_swa(sliding_window_value_length)

    tensors: dict[str, np.ndarray] = {
        "token_embd.weight": np.zeros((64, 32), dtype=np.float16),
        "output_norm.weight": np.zeros((32,), dtype=np.float16),
        "output.weight": np.zeros((32, 64), dtype=np.float16),
    }
    dense_shapes: dict[str, tuple[int, int]] = {}
    for block_index in range(block_count):
        prefix = f"blk.{block_index}"
        tensors[prefix + ".attn_norm.weight"] = np.zeros(
            (32,), dtype=np.float16
        )
        tensors[prefix + ".ffn_norm.weight"] = np.zeros(
            (32,), dtype=np.float16
        )
        dense_shapes.update({
            prefix + ".attn_q.weight": (32, 32),
            prefix + ".attn_k.weight": (32, 32),
            prefix + ".attn_v.weight": (32, 32),
            prefix + ".attn_output.weight": (32, 32),
            prefix + ".ffn_gate.weight": (128, 32),
            prefix + ".ffn_up.weight": (128, 32),
            prefix + ".ffn_down.weight": (32, 128),
        })
    expected_bytes = 0
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor)
        expected_bytes += tensor.nbytes
    quantized_bytes = 0
    for name, shape in dense_shapes.items():
        source = np.zeros(shape, dtype=np.float32)
        encoded = quantize(source, GGMLQuantizationType.Q8_0)
        writer.add_tensor(
            name,
            encoded,
            raw_dtype=GGMLQuantizationType.Q8_0,
        )
        expected_bytes += encoded.nbytes
        quantized_bytes += encoded.nbytes
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {
        "quantized_bytes": quantized_bytes,
        "tensor_bytes": expected_bytes,
    }


class GGUFModelCostTests(unittest.TestCase):
    def test_tensor_lookup_reuses_immutable_manifest_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lookup.gguf"
            write_synthetic_gguf(path)
            manifest = GGUFModelManifestLoader.load(model_id="lookup", path=path)
        before = manifest.to_json()
        index = manifest.tensor_by_id
        self.assertIs(manifest.tensor_by_id, index)
        self.assertEqual(dict(index), {t.tensor_id: t for t in manifest.tensors})
        with self.assertRaises(TypeError):
            index[manifest.tensors[0].tensor_id] = manifest.tensors[0]
        replacement = replace(manifest, tensors=tuple(reversed(manifest.tensors)))
        self.assertIsNot(replacement.tensor_by_id, index)
        self.assertEqual(tuple(replacement.tensor_by_id), tuple(reversed(index)))
        self.assertEqual(manifest.to_json(), before)

    def test_portable_reader_accepts_per_layer_integer_tuple(self) -> None:
        reader = SimpleNamespace(fields={
            "synthetic.attention.head_count_kv": SimpleNamespace(
                data=(0,),
                parts=((8, 8, 1, 8),),
            ),
        })

        self.assertEqual(
            _field_int(reader, "synthetic.attention.head_count_kv"),
            8,
        )

    def test_quantized_bytes_and_generic_operator_dag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unseen-model.gguf"
            expected = write_synthetic_gguf(path)
            manifest = GGUFModelManifestLoader.load(
                model_id="unseen-model-id",
                path=path,
            )

        self.assertEqual(manifest.architecture, "synthetic_transformer")
        self.assertEqual(manifest.block_count, 1)
        self.assertEqual(manifest.tensor_bytes, expected["tensor_bytes"])
        self.assertLess(
            expected["quantized_bytes"],
            7 * 32 * 32 * np.dtype(np.float32).itemsize,
        )
        quantized = [
            tensor for tensor in manifest.tensors
            if tensor.quantization == "Q8_0"
        ]
        self.assertEqual(
            sum(tensor.nbytes for tensor in quantized),
            expected["quantized_bytes"],
        )
        self.assertTrue(all(
            tensor.quantization_block_size == 32
            and tensor.quantization_type_size == 34
            for tensor in quantized
        ))
        self.assertEqual(
            [operator.kind for operator in manifest.operators],
            [
                "embedding",
                "attention_projection",
                "attention",
                "kv_cache",
                "ffn",
                "lm_head",
            ],
        )
        self.assertEqual(
            manifest.operators[0].dependencies,
            (),
        )
        self.assertEqual(
            manifest.operators[-1].dependencies,
            ("layer:0:ffn",),
        )

    def test_attention_kv_and_dense_work_are_geometry_derived(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.gguf"
            write_synthetic_gguf(path)
            manifest = GGUFModelManifestLoader.load("geometry-model", path)

        work = manifest.request_work(input_tokens=4, output_tokens=2)
        attention = work.by_operator_id["layer:0:attention"]
        kv = work.by_operator_id["layer:0:kv_cache"]
        projection = work.by_operator_id["layer:0:attention_projection"]
        expected_attention_flops = (
            4 * 4 * 8 * (1 + 2 + 3 + 4)
            + 4 * 4 * 8 * (4 + 5)
        )
        self.assertEqual(attention.compute_ops, expected_attention_flops)
        self.assertEqual(kv.kv_cache_bytes, 6 * 2 * (8 + 8) * 2)
        self.assertEqual(
            projection.compute_ops,
            4 * (2 * 32 * 32 * 6),
        )
        self.assertEqual(
            projection.weight_bytes,
            sum(
                tensor.nbytes
                for tensor in manifest.tensors
                if tensor.role == "attention_projection"
            ),
        )

    def test_sliding_window_caps_attention_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sliding.gguf"
            write_synthetic_gguf(path, sliding_window=3)
            manifest = GGUFModelManifestLoader.load("sliding-model", path)

        attention = manifest.request_work(
            input_tokens=4, output_tokens=2
        ).by_operator_id["layer:0:attention"]
        context_sum = (1 + 2 + 3 + 3) + (3 + 3)
        self.assertEqual(manifest.sliding_window, 3)
        self.assertEqual(
            attention.compute_ops,
            4 * manifest.head_count * manifest.key_length * context_sum,
        )

    def test_hybrid_attention_preallocated_kv_uses_per_layer_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid.gguf"
            write_synthetic_gguf(
                path,
                block_count=3,
                head_count_kv_by_layer=(2, 1, 2),
                sliding_window=4,
                sliding_window_pattern=(True, False, True),
                sliding_window_key_length=4,
                sliding_window_value_length=4,
            )
            manifest = GGUFModelManifestLoader.load("hybrid-model", path)

        self.assertEqual(manifest.head_count_kv_by_layer, (2, 1, 2))
        self.assertEqual(
            manifest.sliding_window_pattern,
            (True, False, True),
        )
        self.assertEqual(manifest.key_length_swa, 4)
        self.assertEqual(manifest.value_length_swa, 4)
        self.assertEqual(
            sum(
                manifest.preallocated_kv_cache_bytes(
                    f"layer:{index}:kv_cache",
                    context_size=20,
                    parallel=2,
                    sliding_window_padding_tokens=2,
                )
                for index in range(3)
            ),
            2 * (4 * 2 + 2) * 2 * (4 + 4) * 2
            + 20 * 1 * (8 + 8) * 2,
        )

    def test_missing_optional_kv_geometry_uses_generic_attention_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "standard-attention.gguf"
            write_synthetic_gguf(path, explicit_kv_geometry=False)
            manifest = GGUFModelManifestLoader.load(
                "standard-attention-model", path
            )

        self.assertEqual(manifest.head_count_kv, manifest.head_count)
        self.assertEqual(
            manifest.key_length,
            manifest.embedding_length // manifest.head_count,
        )
        self.assertEqual(manifest.value_length, manifest.key_length)

    def test_dependency_free_reader_matches_the_primary_gguf_reader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "portable-reader.gguf"
            write_synthetic_gguf(path, sliding_window=7)
            expected = GGUFModelManifestLoader.load("portable-model", path)
            with mock.patch(
                "research_dev.scheduler._internal.model_manifest._reader_class",
                side_effect=ModuleNotFoundError("synthetic missing dependency"),
            ):
                actual = GGUFModelManifestLoader.load(
                    "portable-model", path
                )

        self.assertEqual(actual.to_json(), expected.to_json())


if __name__ == "__main__":
    unittest.main()
