#!/usr/bin/env python3

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "graph_adapter_v1"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import ubatch_graph_adapter as adapter  # noqa: E402


def tensor_source(
    tensor_id: str,
    name: str,
    shape: list[int],
    nbytes: int,
    tensor_type: str = "F32",
    block_size: int = 1,
) -> dict[str, object]:
    return {
        "tensor_id": tensor_id,
        "name": name,
        "op": "NONE",
        "type": tensor_type,
        "shape": shape,
        "strides": [4, 4 * shape[0]],
        "nbytes": nbytes,
        "buffer_type": "CPU",
        "block_size": block_size,
        "root_tensor_id": tensor_id,
        "root_name": name,
        "root_type": tensor_type,
        "root_shape": shape,
        "root_strides": [4, 4 * shape[0]],
        "root_nbytes": nbytes,
        "root_buffer_type": "CPU",
        "root_block_size": block_size,
        "view_offset": 0,
    }


def matmul_node(
    tensor_id: str,
    name: str,
    weight_id: str,
    weight_name: str,
    weight_shape: list[int],
    output_shape: list[int],
    activation_id: str,
    activation_shape: list[int],
) -> dict[str, object]:
    weight_bytes = weight_shape[0] * weight_shape[1] // 32 * 18
    activation_bytes = 4
    for value in activation_shape:
        activation_bytes *= value
    output_bytes = 4
    for value in output_shape:
        output_bytes *= value
    return {
        "tensor_id": tensor_id,
        "name": name,
        "op": "MUL_MAT",
        "type": "F32",
        "shape": output_shape,
        "strides": [4, 4 * output_shape[0]],
        "nbytes": output_bytes,
        "buffer_type": "CPU",
        "sources": [
            tensor_source(
                weight_id,
                weight_name,
                weight_shape,
                weight_bytes,
                "Q4_0",
                32,
            ),
            tensor_source(
                activation_id,
                activation_id,
                activation_shape,
                activation_bytes,
            ),
        ],
    }


def fixture() -> dict[str, object]:
    m = 8
    k = 3840
    n = 15360
    vocab = 256000
    nodes = [
        matmul_node(
            "up", "ffn_up-0", "w-up", "blk.0.ffn_up.weight",
            [k, n], [n, m], "ffn-input", [k, m],
        ),
        matmul_node(
            "gate", "ffn_gate-0", "w-gate", "blk.0.ffn_gate.weight",
            [k, n], [n, m], "ffn-input", [k, m],
        ),
        matmul_node(
            "down", "ffn_out-0", "w-down", "blk.0.ffn_down.weight",
            [n, k], [k, m], "ffn-gated", [n, m],
        ),
        matmul_node(
            "head", "result_output", "w-head", "output.weight",
            [k, vocab], [vocab, m], "head-input", [k, m],
        ),
        {
            "tensor_id": "norm",
            "name": "attn_norm-0",
            "op": "RMS_NORM",
            "type": "F32",
            "shape": [k, m],
            "strides": [4, 4 * k],
            "nbytes": k * m * 4,
            "buffer_type": "CPU",
            "sources": [tensor_source(
                "hidden", "hidden", [k, m], k * m * 4
            )],
        },
    ]
    return {
        "schema": adapter.INPUT_SCHEMA,
        "status": "PASS",
        "model": {
            "path": "/models/gemma.gguf",
            "architecture": "gemma4",
            "description": "Gemma fixture",
            "n_embd": k,
            "n_layer": 1,
            "n_vocab": vocab,
            "tensor_bytes": 10_000_000_000,
            "parameter_count": 12_000_000_000,
        },
        "context": {
            "n_ctx": 1024,
            "n_batch": 128,
            "n_ubatch": 32,
            "n_seq_max": 8,
            "kv_unified": True,
        },
        "runtime": {
            "requested_device": "CPU",
            "available_devices": ["CPU"],
            "gpu_layers": 0,
            "system_info": "fixture",
        },
        "logical_batch": {
            "decode_requests": 7,
            "decode_context": 32,
            "prefill_tokens": 1,
            "rows": 8,
        },
        "kv_ownership": {
            "owner": "llama_context",
            "before": [],
            "after": [],
        },
        "physical_ubatches": [{
            "index": 0,
            "observed_tokens": m,
            "observed_outputs": m,
            "expected": {
                "index": 0,
                "n_tokens": m,
                "requested_outputs": m,
                "graph_outputs": m,
                "graph_output_floor_applied": False,
                "valid": True,
                "spans": [],
            },
            "nodes": nodes,
        }],
    }


class UbatchGraphAdapterTest(unittest.TestCase):
    def test_dense_ffn_and_head_are_materialized(self) -> None:
        result = adapter.adapt(fixture())
        self.assertEqual(result["schema"], adapter.OUTPUT_SCHEMA)
        self.assertEqual(result["status"], "PASS")
        dense = next(
            row for row in result["operators"]
            if row["family"] == "dense_ffn"
        )
        self.assertEqual(dense["shape"], {"m": 8, "k": 3840, "n": 15360})
        self.assertEqual(dense["compute_ops"], 6 * 8 * 3840 * 15360)
        self.assertEqual(dense["input_bytes"], 8 * 3840 * 4)
        self.assertEqual(dense["output_bytes"], 8 * 3840 * 4)
        self.assertEqual(dense["split_options"][0]["quantum"], 32)
        self.assertEqual(dense["split_options"][0]["merge"], "elementwise_sum")
        self.assertEqual(len(dense["resident_allocation_ids"]), 3)

        head = next(
            row for row in result["operators"]
            if row["family"] == "lm_head"
        )
        self.assertEqual(head["shape"], {"m": 8, "k": 3840, "n": 256000})
        self.assertEqual(head["split_options"][0]["axis"], "vocab_rows")
        self.assertFalse(result["qualification"]["runtime_route"])

    def test_resident_weights_are_deduplicated(self) -> None:
        manifest = fixture()
        duplicate = copy.deepcopy(manifest["physical_ubatches"][0])
        duplicate["index"] = 1
        duplicate["expected"]["index"] = 1
        manifest["physical_ubatches"].append(duplicate)
        manifest["logical_batch"]["rows"] *= 2
        manifest["logical_batch"]["decode_requests"] = 15
        result = adapter.adapt(manifest)
        self.assertEqual(len(result["resident_allocations"]), 4)
        self.assertEqual(result["coverage"]["physical_ubatches"], 2)

    def test_same_weight_name_in_different_buffers_is_not_deduplicated(self) -> None:
        manifest = fixture()
        head_source = manifest["physical_ubatches"][0]["nodes"][3]["sources"][0]
        head_source["root_name"] = "token_embd.weight"
        head_source["name"] = "token_embd.weight"
        duplicate = copy.deepcopy(manifest["physical_ubatches"][0]["nodes"][3])
        duplicate["tensor_id"] = "head-cuda"
        duplicate["name"] = "head-cuda"
        duplicate_source = duplicate["sources"][0]
        duplicate_source["tensor_id"] = "w-head-cuda"
        duplicate_source["root_tensor_id"] = "w-head-cuda"
        duplicate_source["root_name"] = "token_embd.weight"
        duplicate_source["name"] = "token_embd.weight"
        duplicate_source["buffer_type"] = "CUDA0"
        duplicate_source["root_buffer_type"] = "CUDA0"
        manifest["physical_ubatches"][0]["nodes"].append(duplicate)
        result = adapter.adapt(manifest)
        named = [
            row for row in result["resident_allocations"]
            if row["name"] == "token_embd.weight"
        ]
        self.assertEqual(len(named), 2)
        self.assertEqual(
            {row["buffer_type"] for row in named}, {"CPU", "CUDA0"}
        )

    def test_observed_shape_mismatch_fails_closed(self) -> None:
        manifest = fixture()
        manifest["physical_ubatches"][0]["observed_tokens"] = 7
        with self.assertRaises(adapter.GraphAdapterError):
            adapter.adapt(manifest)

    def test_head_projection_requires_observed_weight(self) -> None:
        manifest = fixture()
        manifest["physical_ubatches"][0]["nodes"][3]["sources"] = []
        result = adapter.adapt(manifest)
        self.assertFalse(any(
            row["family"] == "lm_head" for row in result["operators"]
        ))


if __name__ == "__main__":
    unittest.main()
