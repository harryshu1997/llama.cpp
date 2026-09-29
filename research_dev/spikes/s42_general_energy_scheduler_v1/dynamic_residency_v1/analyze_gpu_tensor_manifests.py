#!/usr/bin/env python3
"""Validate exact GPU tensor manifests against measured 4060 Ti residency."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from dynamic_residency_v1.analyze_dual_residency_abba import (  # noqa: E402
    DEFAULT_ROOTS as DEFAULT_ABBA_ROOTS,
    analyze as analyze_abba,
)


DEFAULT_QWEN_MANIFEST = (
    HERE / "results/QWEN_GPU_TENSOR_MANIFEST_V1.json"
)
DEFAULT_GEMMA_MANIFEST = (
    HERE / "results/GEMMA_GPU_TENSOR_MANIFEST_V1.json"
)
DEFAULT_CAPACITY = (
    HERE / "results/RTX4060TI_QWEN15_GEMMA1_CAPACITY_V1/RESULT.json"
)
DEFAULT_ABBA = HERE / "results/DUAL_RESIDENCY_SERVICE_ABBA_V1.json"

SCHEMA = "s42-gpu-tensor-manifest-bundle-v1"
MANIFEST_SCHEMA = "s42-llama-gpu-tensor-manifest-v1"
EXPORTER_SHA256 = (
    "b487349b8f4ee50b55ee5a22f1bb4fe3"
    "86c7f1c313ebcdb2371f8553de8eac30"
)
GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
MIB = 1024 * 1024
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
BLOCK_PATTERN = re.compile(r"^blk\.([0-9]+)\.")

EXPECTED = {
    "qwen": {
        "block_count": 40,
        "gguf": {
            "alignment": 32,
            "data_offset": 5_959_040,
            "tensor_count": 443,
            "tensor_layout_sha256": (
                "d20d658d84c9e1aa218f2e92239010e6"
                "793a7e1ed068870c5d4030587fb22b29"
            ),
        },
        "model": {
            "id": "qwen3-14b-f16-proxy",
            "path": (
                "/home/zhihao/models/"
                "Qwen3-14B-Q4KM-dequant-f16.gguf"
            ),
            "sha256": (
                "d89e9e823744222e595e0b3c8fd5436c"
                "e5d3a6a446fa42492ebce6064dfa9718"
            ),
            "size_bytes": 29_543_423_360,
        },
        "placements": {
            15: {
                "output_layer_on_gpu": True,
                "placement_weight_sha256": (
                    "fa7eb0b1b94c7a8ac7ce6bf85e3de0b8"
                    "17dd25ca4e9fec0748c2e71d5413c0cf"
                ),
                "repeating_layer_ids": list(range(26, 40)),
                "selected_materialized_raw_bytes": 10_804_873_216,
                "selected_tensor_count": 156,
            },
            18: {
                "output_layer_on_gpu": True,
                "placement_weight_sha256": (
                    "1d18b5983acc90b064125fad42db4c43b"
                    "ae9b7bdc2d4c54cd6f0211df1696b6a"
                ),
                "repeating_layer_ids": list(range(23, 40)),
                "selected_materialized_raw_bytes": 12_786_807_808,
                "selected_tensor_count": 189,
            },
        },
    },
    "gemma": {
        "block_count": 48,
        "gguf": {
            "alignment": 32,
            "data_offset": 15_822_368,
            "tensor_count": 667,
            "tensor_layout_sha256": (
                "d652e1759fd6c7f2840dcf2f1885d8e3"
                "e9de376a0afc729ed32e3215cfa261b4"
            ),
        },
        "model": {
            "id": "gemma4-12b-f16-proxy",
            "path": (
                "/home/zhihao/models/"
                "gemma-4-12B-Q40-dequant-f16.gguf"
            ),
            "sha256": (
                "ed76f2183d2d1d65091986033023e6c78"
                "d27f6276c1b0c5826cc92acf73538cf"
            ),
            "size_bytes": 23_832_065_056,
        },
        "placements": {
            0: {
                "output_layer_on_gpu": False,
                "placement_weight_sha256": (
                    "7b91ee36fbff59bf10c585d7a4888a453"
                    "bf6a269a3c9e488f90729e428e1fb1a"
                ),
                "repeating_layer_ids": [],
                "selected_materialized_raw_bytes": 0,
                "selected_tensor_count": 0,
            },
            1: {
                "output_layer_on_gpu": True,
                "placement_weight_sha256": (
                    "3fd0b3c9462f0a87178381da368efb2e"
                    "3434ef0d63234aade1b0f3e1554347ad"
                ),
                "repeating_layer_ids": [],
                "selected_materialized_raw_bytes": 2_013_281_280,
                "selected_tensor_count": 2,
            },
        },
    },
}


class TensorManifestBundleError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TensorManifestBundleError(message)


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
        raise TensorManifestBundleError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TensorManifestBundleError(f"cannot read {path}: {exc}") from exc
    require(type(value) is dict, f"object: {path}")
    return value


def validate_record(value: dict[str, Any], label: str) -> None:
    claimed = value.get("record_sha256")
    unsigned = dict(value)
    unsigned.pop("record_sha256", None)
    require(
        type(claimed) is str
        and SHA_PATTERN.fullmatch(claimed) is not None
        and hashlib.sha256(canonical(unsigned)).hexdigest() == claimed,
        f"{label} record identity",
    )


def validate_manifest(
    value: dict[str, Any], model_key: str
) -> dict[int, dict[str, Any]]:
    expected = EXPECTED[model_key]
    validate_record(value, f"{model_key} manifest")
    require(
        value.get("schema") == MANIFEST_SCHEMA
        and value.get("status") == "EXACT_GGUF_TENSOR_RANGES"
        and value.get("assignment_rule") == {
            "input_layer": "CPU",
            "source": "src/llama-model.cpp:tail-layer-offload",
            "tensor_roles": (
                "src/llama-model-loader.cpp:llm_tensor_layer"
            ),
        }
        and value.get("exporter", {}).get("sha256") == EXPORTER_SHA256
        and value.get("block_count") == expected["block_count"]
        and value.get("model") == expected["model"]
        and value.get("gguf") == expected["gguf"],
        f"{model_key} manifest source and model identity",
    )
    tensors = value.get("selected_tensors")
    require(type(tensors) is list, f"{model_key} selected tensor table")
    tensor_by_name: dict[str, dict[str, Any]] = {}
    ranges: list[tuple[int, int, str]] = []
    alignment = expected["gguf"]["alignment"]
    data_start = expected["gguf"]["data_offset"]
    model_size = expected["model"]["size_bytes"]
    for row in tensors:
        require(type(row) is dict, f"{model_key} tensor row")
        name = row.get("name")
        offset = row.get("data_offset")
        size = row.get("n_bytes")
        shape = row.get("shape")
        digest = row.get("raw_sha256")
        require(
            type(name) is str
            and name
            and name not in tensor_by_name
            and type(offset) is int
            and offset >= data_start
            and offset % alignment == 0
            and type(size) is int
            and size > 0
            and offset + size <= model_size
            and type(shape) is list
            and shape
            and all(type(dimension) is int and dimension > 0 for dimension in shape)
            and type(row.get("tensor_type")) is str
            and SHA_PATTERN.fullmatch(str(digest)) is not None,
            f"{model_key} tensor descriptor",
        )
        tensor_by_name[name] = row
        ranges.append((offset, offset + size, name))
    ordered_ranges = sorted(ranges)
    require(
        all(
            ordered_ranges[index - 1][1] <= ordered_ranges[index][0]
            for index in range(1, len(ordered_ranges))
        ),
        f"{model_key} nonoverlapping tensor ranges",
    )

    placements = value.get("placements")
    require(type(placements) is list, f"{model_key} placements")
    placement_by_layers: dict[int, dict[str, Any]] = {}
    union_names: set[str] = set()
    for placement in placements:
        require(type(placement) is dict, f"{model_key} placement row")
        layers = placement.get("n_gpu_layers")
        require(
            type(layers) is int
            and layers in expected["placements"]
            and layers not in placement_by_layers,
            f"{model_key} placement layer count",
        )
        expected_placement = expected["placements"][layers]
        require(
            placement.get("model_sha256") == expected["model"]["sha256"]
            and all(
                placement.get(key) == expected_value
                for key, expected_value in expected_placement.items()
            ),
            f"{model_key} placement geometry: {layers}",
        )
        claimed = placement.get("placement_weight_sha256")
        unsigned = dict(placement)
        unsigned.pop("placement_weight_sha256", None)
        require(
            hashlib.sha256(canonical(unsigned)).hexdigest() == claimed,
            f"{model_key} placement hash: {layers}",
        )
        entries = placement.get("entries")
        require(
            type(entries) is list
            and len(entries) == placement["selected_tensor_count"]
            and sum(entry.get("n_bytes", -1) for entry in entries)
                == placement["selected_materialized_raw_bytes"],
            f"{model_key} placement entries: {layers}",
        )
        names: list[str] = []
        repeating_layers = set(placement["repeating_layer_ids"])
        for entry in entries:
            require(type(entry) is dict, f"{model_key} placement entry")
            name = entry.get("name")
            layer_id = entry.get("layer_id")
            role = entry.get("materialization_role")
            source = tensor_by_name.get(name)
            require(
                type(name) is str
                and source is not None
                and entry.get("n_bytes") == source["n_bytes"]
                and entry.get("raw_sha256") == source["raw_sha256"]
                and role in {
                    "OUTPUT_LAYER",
                    "REPEATING_LAYER",
                    "TIED_OUTPUT_DUPLICATE",
                },
                f"{model_key} placement tensor binding: {layers}",
            )
            match = BLOCK_PATTERN.match(name)
            if role == "REPEATING_LAYER":
                require(
                    match is not None
                    and type(layer_id) is int
                    and int(match.group(1)) == layer_id
                    and layer_id in repeating_layers,
                    f"{model_key} repeating tensor role: {name}",
                )
            else:
                require(
                    match is None and layer_id is None,
                    f"{model_key} output tensor role: {name}",
                )
            names.append(name)
        require(
            names == sorted(names) and len(names) == len(set(names)),
            f"{model_key} ordered unique placement entries: {layers}",
        )
        union_names.update(names)
        placement_by_layers[layers] = placement
    require(
        set(placement_by_layers) == set(expected["placements"])
        and union_names == set(tensor_by_name),
        f"{model_key} placement and selected tensor coverage",
    )
    return placement_by_layers


def validate_capacity(value: dict[str, Any]) -> None:
    validate_record(value, "capacity")
    require(
        value.get("schema") == "s42-dynamic-gpu-residency-capacity-v1"
        and value.get("status") == "CAPACITY_PASS"
        and value.get("measurement_scope")
            == "CAPACITY_ONLY_NO_ENERGY_OR_SERVICE_CLAIM"
        and value.get("configuration") == {
            "gemma_gpu_layers": 1,
            "gpu_reserve_bytes": 536_870_912,
            "qwen_gpu_layers": 15,
        }
        and value.get("baseline_gpu", {}).get("uuid") == GPU_UUID
        and value.get("cleanup", {}).get("passed") is True
        and value.get("cleanup", {}).get("compute_apps") == [],
        "capacity identity and gates",
    )
    stages = value.get("stages")
    require(
        type(stages) is list
        and len(stages) == 2
        and stages[0].get("allocations", {}).get("offloaded_layers") == 15
        and stages[1].get("allocations", {}).get("offloaded_layers") == 1
        and all(
            stage.get("process_memory", {}).get("swap_bytes") == 0
            for stage in stages
        ),
        "capacity placement stages",
    )


def validate_abba(value: dict[str, Any]) -> None:
    validate_record(value, "service A-B-B-A")
    require(
        value.get("schema") == "s42-dual-residency-service-abba-v1"
        and value.get("status")
            == "REPEATED_SERVICE_DIRECTION_PASS_NO_ADMISSION"
        and value.get("admission", {}).get("eligible") is False
        and value.get("admission", {}).get("dynamic_energy_claim") is None
        and all(value.get("validity_gates", {}).values())
        and value.get("runs", {}).get("control_r1", {}).get(
            "configuration"
        ) == {
            "gemma_gpu_layers": 0,
            "gpu_reserve_bytes": 536_870_912,
            "qwen_gpu_layers": 18,
        }
        and value.get("runs", {}).get("treatment_r1", {}).get(
            "configuration"
        ) == {
            "gemma_gpu_layers": 1,
            "gpu_reserve_bytes": 536_870_912,
            "qwen_gpu_layers": 15,
        },
        "service A-B-B-A identity and placement",
    )


def allocation_observations(
    capacity: dict[str, Any], abba: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    capacity_qwen, capacity_gemma = capacity["stages"]
    controls = [abba["runs"][name] for name in ("control_r1", "control_r2")]
    treatments = [
        abba["runs"][name] for name in ("treatment_r1", "treatment_r2")
    ]
    observed = {
        "gemma_gpu_0": [
            row["stages"]["gemma"]["allocations"]["model_buffer_mib"]
            for row in controls
        ],
        "gemma_gpu_1": [
            capacity_gemma["allocations"]["model_buffer_mib"],
            *(
                row["stages"]["gemma"]["allocations"]["model_buffer_mib"]
                for row in treatments
            ),
        ],
        "qwen_gpu_15": [
            capacity_qwen["allocations"]["model_buffer_mib"],
            *(
                row["stages"]["qwen"]["allocations"]["model_buffer_mib"]
                for row in treatments
            ),
        ],
        "qwen_gpu_18": [
            row["stages"]["qwen"]["allocations"]["model_buffer_mib"]
            for row in controls
        ],
    }
    raw_bytes = {
        "gemma_gpu_0": EXPECTED["gemma"]["placements"][0][
            "selected_materialized_raw_bytes"
        ],
        "gemma_gpu_1": EXPECTED["gemma"]["placements"][1][
            "selected_materialized_raw_bytes"
        ],
        "qwen_gpu_15": EXPECTED["qwen"]["placements"][15][
            "selected_materialized_raw_bytes"
        ],
        "qwen_gpu_18": EXPECTED["qwen"]["placements"][18][
            "selected_materialized_raw_bytes"
        ],
    }
    output = {}
    for name, values in observed.items():
        expected_mib = raw_bytes[name] / MIB
        maximum_error = max(abs(value - expected_mib) for value in values)
        require(maximum_error <= 0.02, f"model buffer binding: {name}")
        output[name] = {
            "logged_model_buffer_mib": values,
            "maximum_log_rounding_error_mib": maximum_error,
            "raw_tensor_bytes": raw_bytes[name],
        }
    return output


def analyze(
    qwen_manifest: dict[str, Any],
    gemma_manifest: dict[str, Any],
    capacity: dict[str, Any],
    abba: dict[str, Any],
    *,
    evidence: dict[str, str],
) -> dict[str, Any]:
    qwen = validate_manifest(qwen_manifest, "qwen")
    gemma = validate_manifest(gemma_manifest, "gemma")
    validate_capacity(capacity)
    validate_abba(abba)

    qwen_15_names = {row["name"] for row in qwen[15]["entries"]}
    qwen_18_names = {row["name"] for row in qwen[18]["entries"]}
    evicted_names = qwen_18_names - qwen_15_names
    evicted_entries = [
        row for row in qwen[18]["entries"] if row["name"] in evicted_names
    ]
    evicted_layers = sorted({row["layer_id"] for row in evicted_entries})
    evicted_raw_bytes = sum(row["n_bytes"] for row in evicted_entries)
    require(
        qwen_15_names < qwen_18_names
        and len(evicted_entries) == 33
        and evicted_layers == [23, 24, 25]
        and all(
            row["materialization_role"] == "REPEATING_LAYER"
            for row in evicted_entries
        )
        and evicted_raw_bytes == 1_981_934_592,
        "Qwen 18-to-15 exact tensor difference",
    )
    require(gemma[0]["entries"] == [], "Gemma zero-layer manifest")
    gemma_roles = {
        row["name"]: row["materialization_role"]
        for row in gemma[1]["entries"]
    }
    require(
        gemma_roles == {
            "output_norm.weight": "OUTPUT_LAYER",
            "token_embd.weight": "TIED_OUTPUT_DUPLICATE",
        },
        "Gemma output-layer tensor roles",
    )

    allocation = allocation_observations(capacity, abba)
    capacity_qwen, capacity_gemma = capacity["stages"]
    measured_incremental_gpu_bytes = (
        capacity_gemma["gpu"]["memory_used_bytes"]
        - capacity_qwen["gpu"]["memory_used_bytes"]
    )
    require(
        measured_incremental_gpu_bytes == 2_444_230_656,
        "measured Gemma incremental GPU allocation",
    )
    reserve_bytes = capacity["configuration"]["gpu_reserve_bytes"]
    control_minimum_free = min(
        abba["runs"][name]["gpu"]["minimum_free_bytes"]
        for name in ("control_r1", "control_r2")
    )
    stageable_bytes = max(0, control_minimum_free - reserve_bytes)
    staging_shortfall = max(
        0, measured_incremental_gpu_bytes - stageable_bytes
    )
    final_free_beyond_reserve = (
        capacity_gemma["gpu"]["memory_free_bytes"] - reserve_bytes
    )
    qwen_15_stageable_bytes = (
        capacity_qwen["gpu"]["memory_free_bytes"] - reserve_bytes
    )
    gemma_after_qwen_15_margin = (
        qwen_15_stageable_bytes - measured_incremental_gpu_bytes
    )
    require(
        final_free_beyond_reserve == 180_355_072
        and stageable_bytes == 205_520_896
        and staging_shortfall == 2_238_709_760,
        "atomic staging capacity arithmetic",
    )
    require(
        qwen_15_stageable_bytes == 2_624_585_728
        and gemma_after_qwen_15_margin == 180_355_072,
        "two-epoch staging capacity arithmetic",
    )

    gemma_raw_bytes = gemma[1]["selected_materialized_raw_bytes"]
    output: dict[str, Any] = {
        "atomic_staging": {
            "atomic_stage_before_evict_fits": False,
            "control_conservative_minimum_free_bytes": (
                control_minimum_free
            ),
            "drain_or_evict_before_stage_required": True,
            "final_measured_placement_fits": True,
            "final_placement_free_beyond_reserve_bytes": (
                final_free_beyond_reserve
            ),
            "gemma_measured_incremental_gpu_bytes": (
                measured_incremental_gpu_bytes
            ),
            "gemma_after_qwen15_atomic_stage_fits": True,
            "gemma_after_qwen15_staging_margin_bytes": (
                gemma_after_qwen_15_margin
            ),
            "gpu_reserve_bytes": reserve_bytes,
            "required_transition_mode": (
                "DRAIN_OR_EVICT_BEFORE_STAGE_WITH_READY_FALLBACK"
            ),
            "stageable_bytes_beyond_reserve": stageable_bytes,
            "staging_shortfall_bytes": staging_shortfall,
            "qwen15_stageable_bytes_beyond_reserve": (
                qwen_15_stageable_bytes
            ),
            "required_transition_sequence": [
                {
                    "from": "qwen18+gemma0",
                    "mode": (
                        "DRAIN_OR_EVICT_BEFORE_STAGE_WITH_READY_FALLBACK"
                    ),
                    "step": 1,
                    "to": "qwen15+gemma0",
                },
                {
                    "from": "qwen15+gemma0",
                    "mode": "ATOMIC_STAGE_BEFORE_EVICT",
                    "step": 2,
                    "to": "qwen15+gemma1",
                },
            ],
        },
        "claim_gates": {
            "dynamic_energy_admission": False,
            "exact_gpu_tensor_slice_manifests": True,
            "measured_atomic_stage_before_evict": False,
            "measured_final_placement_capacity": True,
            "measured_second_epoch_atomic_staging_capacity": True,
            "runtime_model_buffer_bound_to_raw_tensor_bytes": True,
            "service_abba_revalidated_from_raw_runs": True,
        },
        "evidence": evidence,
        "manifest_records": {
            "gemma": gemma_manifest["record_sha256"],
            "qwen": qwen_manifest["record_sha256"],
        },
        "placements": {
            "gemma_gpu_0": {
                key: deepcopy(gemma[0][key])
                for key in (
                    "n_gpu_layers",
                    "placement_weight_sha256",
                    "selected_materialized_raw_bytes",
                    "selected_tensor_count",
                )
            },
            "gemma_gpu_1": {
                key: deepcopy(gemma[1][key])
                for key in (
                    "n_gpu_layers",
                    "placement_weight_sha256",
                    "selected_materialized_raw_bytes",
                    "selected_tensor_count",
                )
            },
            "qwen_gpu_15": {
                key: deepcopy(qwen[15][key])
                for key in (
                    "n_gpu_layers",
                    "placement_weight_sha256",
                    "selected_materialized_raw_bytes",
                    "selected_tensor_count",
                )
            },
            "qwen_gpu_18": {
                key: deepcopy(qwen[18][key])
                for key in (
                    "n_gpu_layers",
                    "placement_weight_sha256",
                    "selected_materialized_raw_bytes",
                    "selected_tensor_count",
                )
            },
        },
        "runtime_allocation_binding": allocation,
        "schema": SCHEMA,
        "status": "EXACT_MANIFEST_PASS_DIRECT_ATOMIC_STAGING_BLOCKED",
        "transition_slice": {
            "from": "qwen18+gemma0",
            "gemma_added_raw_tensor_bytes": gemma_raw_bytes,
            "net_added_raw_tensor_bytes": (
                gemma_raw_bytes - evicted_raw_bytes
            ),
            "qwen_evicted_layer_ids": evicted_layers,
            "qwen_evicted_raw_tensor_bytes": evicted_raw_bytes,
            "qwen_evicted_tensor_count": len(evicted_entries),
            "to": "qwen15+gemma1",
        },
    }
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def build(
    qwen_path: Path = DEFAULT_QWEN_MANIFEST,
    gemma_path: Path = DEFAULT_GEMMA_MANIFEST,
    capacity_path: Path = DEFAULT_CAPACITY,
    abba_path: Path = DEFAULT_ABBA,
) -> dict[str, Any]:
    checked_abba = read_object(abba_path)
    regenerated_abba = analyze_abba(DEFAULT_ABBA_ROOTS)
    require(
        checked_abba == regenerated_abba,
        "checked A-B-B-A differs from raw-run analysis",
    )
    paths = {
        "capacity_file_sha256": capacity_path,
        "gemma_manifest_file_sha256": gemma_path,
        "qwen_manifest_file_sha256": qwen_path,
        "service_abba_file_sha256": abba_path,
    }
    return analyze(
        read_object(qwen_path),
        read_object(gemma_path),
        read_object(capacity_path),
        checked_abba,
        evidence={name: sha256(path) for name, path in paths.items()},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-manifest", type=Path, default=DEFAULT_QWEN_MANIFEST
    )
    parser.add_argument(
        "--gemma-manifest", type=Path, default=DEFAULT_GEMMA_MANIFEST
    )
    parser.add_argument("--capacity", type=Path, default=DEFAULT_CAPACITY)
    parser.add_argument("--abba", type=Path, default=DEFAULT_ABBA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    try:
        value = build(
            args.qwen_manifest,
            args.gemma_manifest,
            args.capacity,
            args.abba,
        )
        args.output.write_bytes(canonical(value))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"GPU tensor manifest analysis failed: {exc}\n")
    print(json.dumps({
        "record_sha256": value["record_sha256"],
        "staging_shortfall_bytes": value["atomic_staging"][
            "staging_shortfall_bytes"
        ],
        "status": value["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
