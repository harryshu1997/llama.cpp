#!/usr/bin/env python3
"""Convert a captured llama.cpp physical ubatch graph into placement facts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


INPUT_SCHEMA = "s42-llama-ubatch-graph-v1"
OUTPUT_SCHEMA = "s42-llama-placement-graph-v1"
MATMUL_OPS = {"MUL_MAT", "MUL_MAT_ID"}
METADATA_OPS = {"NONE", "RESHAPE", "VIEW", "PERMUTE", "TRANSPOSE"}
COPY_OPS = {"CPY", "DUP", "CONT"}
WEIGHT_SUFFIXES = (".weight", ".bias", ".scale")
LAYER_PATTERN = re.compile(r"(?:^|\.)blk\.(\d+)\.")


class GraphAdapterError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GraphAdapterError(message)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def object_value(name: str, value: object) -> Mapping[str, Any]:
    require(type(value) is dict, f"{name} must be an object")
    return value


def list_value(name: str, value: object) -> list[Any]:
    require(type(value) is list, f"{name} must be a list")
    return value


def string_value(name: str, value: object) -> str:
    require(type(value) is str and bool(value), f"{name} must be a string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise GraphAdapterError(f"{name} must be ASCII") from error
    return value


def integer_value(name: str, value: object, minimum: int = 0) -> int:
    require(
        type(value) is int and value >= minimum,
        f"{name} must be an integer >= {minimum}",
    )
    return value


def product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def shape_value(name: str, value: object) -> tuple[int, ...]:
    raw = list_value(name, value)
    require(bool(raw), f"{name} must not be empty")
    return tuple(integer_value(name, item, 0) for item in raw)


def is_weight_name(name: str) -> bool:
    return name.endswith(WEIGHT_SUFFIXES)


def source_root_name(source: Mapping[str, Any]) -> str:
    value = source.get("root_name") or source.get("name")
    return string_value("source root name", value)


def weight_source(node: Mapping[str, Any]) -> Mapping[str, Any] | None:
    sources = list_value("node sources", node.get("sources"))
    for source_raw in sources:
        source = object_value("node source", source_raw)
        name = source.get("root_name") or source.get("name")
        if type(name) is str and is_weight_name(name):
            return source
    return None


def activation_sources(node: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = []
    for source_raw in list_value("node sources", node.get("sources")):
        source = object_value("node source", source_raw)
        name = source.get("root_name") or source.get("name")
        if type(name) is not str or not is_weight_name(name):
            result.append(source)
    return result


def layer_from_weight(name: str) -> int:
    match = LAYER_PATTERN.search(name)
    return int(match.group(1)) if match else -1


def role_from_weight(name: str) -> str:
    roles = (
        ("ffn_gate_up_exps", "moe_gate_up"),
        ("ffn_gate_exps", "moe_gate"),
        ("ffn_up_exps", "moe_up"),
        ("ffn_down_exps", "moe_down"),
        ("ffn_gate_up", "ffn_gate_up"),
        ("ffn_gate", "ffn_gate"),
        ("ffn_up", "ffn_up"),
        ("ffn_down", "ffn_down"),
        ("attn_qkv", "attn_qkv"),
        ("attn_q", "attn_q"),
        ("attn_k", "attn_k"),
        ("attn_v", "attn_v"),
        ("attn_output", "attn_output"),
    )
    for needle, role in roles:
        if f".{needle}." in name:
            return role
    if name.endswith("output.weight") or name.endswith("token_embd.weight"):
        return "lm_head"
    return "generic"


def allocation_from_source(
    model_key: str, source: Mapping[str, Any]
) -> dict[str, object]:
    name = source_root_name(source)
    tensor_type = string_value("source root type", source.get("root_type"))
    shape = shape_value("source root shape", source.get("root_shape"))
    buffer_type = string_value(
        "source root buffer", source.get("root_buffer_type")
    )
    nbytes = integer_value("source root bytes", source.get("root_nbytes"), 1)
    identity = canonical({
        "model": model_key,
        "name": name,
        "type": tensor_type,
        "shape": shape,
        "buffer": buffer_type,
    })
    return {
        "allocation_id": "alloc:" + hashlib.sha256(identity).hexdigest()[:24],
        "buffer_type": buffer_type,
        "bytes": nbytes,
        "layout_id": "ggml:{}:{}".format(
            tensor_type, "x".join(str(item) for item in shape)
        ),
        "name": name,
        "shape": list(shape),
        "type": tensor_type,
    }


def matmul_shape(
    node: Mapping[str, Any], source: Mapping[str, Any]
) -> tuple[int, int, int]:
    weight_shape = shape_value("weight shape", source.get("root_shape"))
    output_shape = shape_value("matmul output shape", node.get("shape"))
    require(len(weight_shape) == 2, "matmul weight must be two dimensional")
    k, n = weight_shape
    require(k > 0 and n > 0, "matmul dimensions must be positive")
    output_elements = product(output_shape)
    require(output_elements % n == 0, "matmul output shape mismatch")
    return output_elements // n, k, n


def node_bytes(node: Mapping[str, Any]) -> int:
    return integer_value("node bytes", node.get("nbytes"))


def source_bytes(source: Mapping[str, Any]) -> int:
    return integer_value("source bytes", source.get("nbytes"))


def node_id(node: Mapping[str, Any]) -> str:
    return string_value("node tensor id", node.get("tensor_id"))


def external_inputs(
    nodes: Iterable[Mapping[str, Any]], claimed: set[str]
) -> list[str]:
    result = set()
    for node in nodes:
        for source in activation_sources(node):
            source_id = string_value("source tensor id", source.get("tensor_id"))
            if source_id not in claimed:
                result.add(source_id)
    return sorted(result)


def model_key(model: Mapping[str, Any]) -> str:
    identity = {
        "architecture": model.get("architecture"),
        "description": model.get("description"),
        "parameter_count": model.get("parameter_count"),
        "tensor_bytes": model.get("tensor_bytes"),
        "path": model.get("path"),
    }
    return "model:" + hashlib.sha256(canonical(identity)).hexdigest()[:24]


def dense_ffn_operator(
    graph_index: int,
    layer: int,
    role_nodes: Mapping[str, Mapping[str, Any]],
    model_identity: str,
    allocations: Mapping[str, dict[str, object]],
) -> dict[str, object] | None:
    if "ffn_down" not in role_nodes:
        return None
    up_roles = [role for role in ("ffn_up", "ffn_gate", "ffn_gate_up") if role in role_nodes]
    if "ffn_up" not in role_nodes and "ffn_gate_up" not in role_nodes:
        return None

    ordered_roles = up_roles + ["ffn_down"]
    nodes = [role_nodes[role] for role in ordered_roles]
    shapes = []
    weight_allocations: list[Mapping[str, object]] = []
    for node in nodes:
        source = weight_source(node)
        require(source is not None, "dense FFN matmul lacks a weight")
        shapes.append(matmul_shape(node, source))
        allocation = allocation_from_source(model_identity, source)
        allocation_id = string_value(
            "allocation id", allocation["allocation_id"]
        )
        require(allocation_id in allocations, "resident allocation is missing")
        weight_allocations.append(allocations[allocation_id])

    m = shapes[0][0]
    up_k = shapes[0][1]
    up_n = shapes[0][2]
    require(all(shape[0] == m for shape in shapes), "dense FFN M mismatch")
    down_shape = shapes[-1]
    if "ffn_gate_up" in role_nodes and "ffn_up" not in role_nodes:
        require(up_n % 2 == 0, "fused FFN width must be even")
        n = up_n // 2
    else:
        n = up_n
    require(
        down_shape[1] == n and down_shape[2] == up_k,
        "dense FFN up/down shape mismatch",
    )

    compute_ops = sum(2 * mm * kk * nn for mm, kk, nn in shapes)
    input_candidates = activation_sources(nodes[0])
    input_bytes = max((source_bytes(source) for source in input_candidates), default=0)
    output_bytes = node_bytes(nodes[-1])
    block_sizes = []
    for node in nodes:
        source = weight_source(node)
        block_sizes.append(integer_value(
            "weight block size", source.get("root_block_size"), 1
        ))
    quantum = math.lcm(*block_sizes)
    require(n > quantum, "dense FFN is too narrow to split")
    claimed = {node_id(node) for node in nodes}
    unique_allocations = {
        string_value("allocation id", row["allocation_id"]): row
        for row in weight_allocations
    }
    return {
        "operator_id": f"g{graph_index}:l{layer}:dense_ffn",
        "family": "dense_ffn",
        "layer": layer,
        "shape": {"m": m, "k": up_k, "n": n},
        "compute_ops": compute_ops,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "memory_bytes": sum(
            integer_value("allocation bytes", row["bytes"], 1)
            for row in unique_allocations.values()
        ) + input_bytes + output_bytes,
        "node_ids": [node_id(node) for node in nodes],
        "input_tensor_ids": external_inputs(nodes, claimed),
        "output_tensor_id": node_id(nodes[-1]),
        "resident_allocation_ids": sorted(unique_allocations),
        "split_options": [{
            "axis": "ffn_columns",
            "total": n,
            "quantum": quantum,
            "minimum": quantum,
            "maximum": n - quantum,
            "host_partition": "prefix",
            "remote_partition": "suffix",
            "merge": "elementwise_sum",
            "upload_bytes": input_bytes,
            "download_bytes": output_bytes,
            "status": "structurally_legal",
        }],
    }


def matmul_operator(
    graph_index: int,
    node: Mapping[str, Any],
    allocation: Mapping[str, object],
) -> dict[str, object]:
    source = weight_source(node)
    require(source is not None, "matmul lacks a weight")
    m, k, n = matmul_shape(node, source)
    role = role_from_weight(source_root_name(source))
    layer = layer_from_weight(source_root_name(source))
    input_bytes = max(
        (source_bytes(item) for item in activation_sources(node)), default=0
    )
    if role in {"attn_q", "attn_k", "attn_v", "attn_qkv"}:
        axis = "attention_output_columns"
        merge = "concatenate"
        quantum = 1
        status = "requires_head_alignment"
    elif role == "attn_output":
        axis = "attention_reduction_k"
        merge = "elementwise_sum"
        quantum = integer_value(
            "weight block size", source.get("root_block_size"), 1
        )
        status = "structurally_legal"
    elif role == "lm_head":
        axis = "vocab_rows"
        merge = "concatenate_logits"
        quantum = 1
        status = "structurally_legal"
    else:
        axis = "output_columns"
        merge = "concatenate"
        quantum = 1
        status = "requires_operator_semantics"
    total = k if axis.endswith("reduction_k") else n
    split_options = []
    if total > quantum:
        split_options.append({
            "axis": axis,
            "total": total,
            "quantum": quantum,
            "minimum": quantum,
            "maximum": total - quantum,
            "merge": merge,
            "upload_bytes": input_bytes,
            "download_bytes": node_bytes(node),
            "status": status,
        })
    return {
        "operator_id": f"g{graph_index}:l{layer}:{role}:{node_id(node)}",
        "family": role,
        "layer": layer,
        "shape": {"m": m, "k": k, "n": n},
        "compute_ops": 2 * m * k * n,
        "input_bytes": input_bytes,
        "output_bytes": node_bytes(node),
        "memory_bytes": integer_value("allocation bytes", allocation["bytes"], 1) + input_bytes + node_bytes(node),
        "node_ids": [node_id(node)],
        "input_tensor_ids": external_inputs([node], {node_id(node)}),
        "output_tensor_id": node_id(node),
        "resident_allocation_ids": [allocation["allocation_id"]],
        "split_options": split_options,
    }


def moe_operator(
    graph_index: int,
    layer: int,
    nodes: list[Mapping[str, Any]],
    model_identity: str,
    allocations: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    allocation_rows: list[Mapping[str, object]] = []
    experts = None
    input_bytes = 0
    output_bytes = 0
    for node in nodes:
        source = weight_source(node)
        require(source is not None, "MoE matmul lacks a weight")
        shape = shape_value("MoE weight shape", source.get("root_shape"))
        require(len(shape) >= 3, "MoE weight has no expert dimension")
        experts = shape[2] if experts is None else experts
        require(experts == shape[2], "MoE expert dimension mismatch")
        allocation = allocation_from_source(model_identity, source)
        allocation_id = string_value(
            "allocation id", allocation["allocation_id"]
        )
        require(allocation_id in allocations, "resident allocation is missing")
        allocation_rows.append(allocations[allocation_id])
        input_bytes = max(
            input_bytes,
            max((source_bytes(item) for item in activation_sources(node)), default=0),
        )
        output_bytes = max(output_bytes, node_bytes(node))
    claimed = {node_id(node) for node in nodes}
    unique_allocations = {
        string_value("allocation id", row["allocation_id"]): row
        for row in allocation_rows
    }
    return {
        "operator_id": f"g{graph_index}:l{layer}:expert_ffn",
        "family": "expert_ffn",
        "layer": layer,
        "shape": {"experts": experts, "active_experts": None},
        "compute_ops": None,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "memory_bytes": sum(
            integer_value("allocation bytes", row["bytes"], 1)
            for row in unique_allocations.values()
        ) + input_bytes + output_bytes,
        "node_ids": [node_id(node) for node in nodes],
        "input_tensor_ids": external_inputs(nodes, claimed),
        "output_tensor_id": node_id(nodes[-1]),
        "resident_allocation_ids": sorted(unique_allocations),
        "split_options": [{
            "axis": "experts",
            "total": experts,
            "quantum": 1,
            "minimum": 1,
            "maximum": experts - 1,
            "merge": "router_weighted_sum",
            "upload_bytes": input_bytes,
            "download_bytes": output_bytes,
            "status": "requires_runtime_routing",
        }],
    }


def adapt(
    manifest: Mapping[str, Any], manifest_bytes: bytes | None = None
) -> dict[str, object]:
    require(manifest.get("schema") == INPUT_SCHEMA, "input schema mismatch")
    require(manifest.get("status") == "PASS", "input capture did not pass")
    model = object_value("model", manifest.get("model"))
    key = model_key(model)
    physical = list_value(
        "physical_ubatches", manifest.get("physical_ubatches")
    )
    require(bool(physical), "physical_ubatches must not be empty")

    allocations: dict[str, dict[str, object]] = {}
    validated_graphs = []
    all_operators = []
    local_only = []
    claimed_nodes: set[tuple[int, str]] = set()
    total_tokens = 0
    total_requested_outputs = 0

    for expected_index, graph_raw in enumerate(physical):
        graph = object_value("physical ubatch", graph_raw)
        index = integer_value("physical ubatch index", graph.get("index"))
        require(index == expected_index, "physical ubatch indices are not contiguous")
        expected = object_value("physical ubatch expected", graph.get("expected"))
        require(expected.get("valid") is True, "physical ubatch is not valid")
        require(
            expected.get("index") == index,
            "physical ubatch expected index mismatch",
        )
        n_tokens = integer_value("physical ubatch tokens", expected.get("n_tokens"), 1)
        requested_outputs = integer_value(
            "physical ubatch requested outputs",
            expected.get("requested_outputs"),
        )
        graph_outputs = integer_value(
            "physical ubatch graph outputs", expected.get("graph_outputs"), 1
        )
        require(graph.get("observed_tokens") == n_tokens, "observed token mismatch")
        require(
            graph.get("observed_outputs") == graph_outputs,
            "observed output mismatch",
        )
        require(
            expected.get("graph_output_floor_applied")
            is (requested_outputs == 0),
            "graph output floor flag mismatch",
        )
        total_tokens += n_tokens
        total_requested_outputs += requested_outputs
        nodes_raw = list_value("physical ubatch nodes", graph.get("nodes"))
        require(bool(nodes_raw), "physical ubatch has no nodes")
        nodes = [object_value("graph node", item) for item in nodes_raw]
        ids = [node_id(node) for node in nodes]
        require(len(ids) == len(set(ids)), "duplicate graph tensor id")

        for node in nodes:
            source = weight_source(node)
            if source is None:
                continue
            row = allocation_from_source(key, source)
            allocation_id = string_value(
                "allocation id", row["allocation_id"]
            )
            if allocation_id in allocations:
                require(
                    allocations[allocation_id] == row,
                    "resident allocation changed",
                )
            else:
                allocations[allocation_id] = row

        dense: dict[int, dict[str, Mapping[str, Any]]] = {}
        moe: dict[int, list[Mapping[str, Any]]] = {}
        matmuls = []
        for node in nodes:
            if node.get("op") not in MATMUL_OPS:
                continue
            source = weight_source(node)
            if source is None:
                continue
            name = source_root_name(source)
            role = role_from_weight(name)
            layer = layer_from_weight(name)
            if role.startswith("moe_"):
                moe.setdefault(layer, []).append(node)
            elif role in {"ffn_up", "ffn_gate", "ffn_gate_up", "ffn_down"}:
                dense.setdefault(layer, {})[role] = node
            else:
                matmuls.append(node)

        operators = []
        for layer, role_nodes in sorted(dense.items()):
            operator = dense_ffn_operator(
                index, layer, role_nodes, key, allocations
            )
            if operator is not None:
                operators.append(operator)
                claimed_nodes.update((index, item) for item in operator["node_ids"])
        for layer, group in sorted(moe.items()):
            operator = moe_operator(index, layer, group, key, allocations)
            operators.append(operator)
            claimed_nodes.update((index, item) for item in operator["node_ids"])
        for node in matmuls:
            source = weight_source(node)
            allocation = allocation_from_source(key, source)
            operator = matmul_operator(
                index,
                node,
                allocations[string_value(
                    "allocation id", allocation["allocation_id"]
                )],
            )
            operators.append(operator)
            claimed_nodes.add((index, node_id(node)))

        for node in nodes:
            current_id = node_id(node)
            op = string_value("node op", node.get("op"))
            if (
                (index, current_id) not in claimed_nodes
                and op not in METADATA_OPS
                and op not in COPY_OPS
            ):
                local_only.append({
                    "graph_index": index,
                    "node_id": current_id,
                    "name": node.get("name", ""),
                    "op": op,
                    "reason": "not_an_independently_splittable_profiled_operator",
                })
        operators.sort(key=lambda item: item["operator_id"])
        all_operators.extend(operators)
        validated_graphs.append({
            "index": index,
            "n_tokens": n_tokens,
            "requested_outputs": requested_outputs,
            "graph_outputs": graph_outputs,
            "spans": expected.get("spans"),
            "node_count": len(nodes),
            "operator_ids": [item["operator_id"] for item in operators],
        })

    logical_batch = object_value("logical batch", manifest.get("logical_batch"))
    logical_rows = integer_value("logical batch rows", logical_batch.get("rows"), 1)
    decode_requests = integer_value(
        "logical decode requests", logical_batch.get("decode_requests"), 1
    )
    require(total_tokens == logical_rows, "physical token total mismatch")
    require(
        total_requested_outputs == decode_requests + 1,
        "physical requested output total mismatch",
    )

    input_bytes = canonical(manifest) if manifest_bytes is None else manifest_bytes
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "PASS",
        "manifest_sha256": "sha256:" + hashlib.sha256(input_bytes).hexdigest(),
        "model": dict(model),
        "model_key": key,
        "context": dict(object_value("context", manifest.get("context"))),
        "runtime": dict(object_value("runtime", manifest.get("runtime"))),
        "logical_batch": dict(logical_batch),
        "kv_ownership": dict(
            object_value("KV ownership", manifest.get("kv_ownership"))
        ),
        "physical_ubatches": validated_graphs,
        "resident_allocations": sorted(
            allocations.values(), key=lambda item: item["allocation_id"]
        ),
        "operators": all_operators,
        "local_only_nodes": local_only,
        "coverage": {
            "physical_ubatches": len(validated_graphs),
            "placement_operators": len(all_operators),
            "split_options": sum(
                len(item["split_options"]) for item in all_operators
            ),
            "local_only_nodes": len(local_only),
        },
        "qualification": {
            "graph_observed": True,
            "placement_bound": False,
            "energy_bound": False,
            "runtime_route": False,
            "next_gate": "compile and validate an epoch-bound route",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.input.is_file(), "input manifest does not exist")
    require(not args.output.exists(), "output already exists")
    manifest_bytes = args.input.read_bytes()
    manifest = json.loads(manifest_bytes.decode("ascii"))
    result = adapt(object_value("manifest", manifest), manifest_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GraphAdapterError as error:
        print(f"S42_GRAPH_ADAPTER_ERROR: {error}")
        raise SystemExit(2)
