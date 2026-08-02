#!/usr/bin/env python3
"""S41 desktop-smoke authority for the Gemma/Qwen model pair."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from executor_bundle import validate_executor_bundle
from phone_gateway import (
    MAX_COMMAND_BYTES,
    canonical_bytes,
    exact_keys,
    integer,
    require,
    sha256_text,
    strict_json_loads,
    string,
)


AUTHORITY_SCHEMA = "s41-desktop-smoke-authority-v1"
AUTHORITY_SCOPE = "DESKTOP_SMOKE_ONLY"
CONTRACT_RELATIVE = "experiment/EXPERIMENT_CONTRACT.json"
CONTRACT_VALUE = {
    "gpu": {
        "memory_total_mib": 16380,
        "name": "NVIDIA GeForce RTX 4060 Ti",
        "uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
    },
    "models": {
        "gemma-4-12b-it-q8_0": {
            "bytes": 12669645856,
            "quantization": "Q8_0",
            "sha256":
                "7b56cbd0e0d96d5c8d7df9c21b39d67264e58cbabf7a96d4681eff8b3d492848",
        },
        "qwen3-14b-q4_k_m": {
            "bytes": 9001752960,
            "quantization": "Q4_K_M",
            "sha256":
                "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
        },
    },
    "pair": [
        "gemma-4-12b-it-q8_0",
        "qwen3-14b-q4_k_m",
    ],
    "runtime": {
        "llama_server": {
            "bytes": 17904,
            "path":
                "/home/zhihao/llama.cpp-s40/build-s40-cuda/bin/llama-server",
            "sha256":
                "f890165ce1f89d3084bc108d05b42e5ca4e1a0f25aa40fe7786d39ab87562d6a",
            "version": "9875 (7d1926dff)",
        },
        "nvidia_smi": {
            "bytes": 1259616,
            "path": "/usr/bin/nvidia-smi",
            "sha256":
                "dd0cbc1a839dae1cfadb5ba1ffb8e3bfed99ddd8f3d1dca8e986d68ce7d0515c",
        },
    },
    "schema": "s41-gemma-qwen-desktop-smoke-contract-v1",
    "serving": {
        "batch_size": 2048,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "context_size": 4096,
        "continuous_batching": True,
        "flash_attention": True,
        "minimum_free_device_memory_mib": 512,
        "n_gpu_layers": "99",
        "parallel_slots": 8,
        "physical_ubatch_size": 512,
        "split_mode": "none",
    },
    "status": "FROZEN_BEFORE_GEMMA_B1_B8",
}
CONTRACT_BYTES = canonical_bytes(CONTRACT_VALUE)
CONTRACT_SHA256 = hashlib.sha256(CONTRACT_BYTES).hexdigest()
EXPECTED_PAIR = tuple(CONTRACT_VALUE["pair"])
EXPECTED_MODELS = CONTRACT_VALUE["models"]
EXPECTED_GPU = CONTRACT_VALUE["gpu"]
EXPECTED_RUNTIME = CONTRACT_VALUE["runtime"]
EXPECTED_SERVING = CONTRACT_VALUE["serving"]


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def validate_contract_value(value: Any) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "gpu",
            "models",
            "pair",
            "runtime",
            "schema",
            "serving",
            "status",
        },
        "S41 desktop smoke contract",
    )
    require(
        value["schema"] == CONTRACT_VALUE["schema"],
        "S41 desktop smoke contract schema",
    )
    require(
        value["status"] == CONTRACT_VALUE["status"],
        "S41 desktop smoke contract status",
    )
    require(value["pair"] == CONTRACT_VALUE["pair"], "S41 model pair")
    require(value["models"] == EXPECTED_MODELS, "S41 model bindings")
    require(value["gpu"] == EXPECTED_GPU, "S41 GPU binding")
    require(value["runtime"] == EXPECTED_RUNTIME, "S41 runtime binding")
    require(value["serving"] == EXPECTED_SERVING, "S41 serving binding")
    return value


def validate_desktop_smoke_authority(
    value: Any,
    *,
    expected_model_id: str,
    expected_model_path: str,
    executor_bundle: Path,
    executor_bundle_manifest_sha256: str,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "contract_sha256",
            "gpu_uuid",
            "model_bytes",
            "model_id",
            "model_path",
            "model_sha256",
            "phase",
            "phase_id",
            "schema",
            "scope",
        },
        "S41 desktop smoke authority",
    )
    require(value["schema"] == AUTHORITY_SCHEMA, "S41 authority schema")
    require(value["scope"] == AUTHORITY_SCOPE, "S41 authority scope")
    require(value["phase"] == "DESKTOP_SMOKE", "S41 authority phase")
    phase_id = string(value["phase_id"], "S41 authority phase ID")
    require(value["model_id"] == expected_model_id, "S41 authority model")
    require(
        value["contract_sha256"] == CONTRACT_SHA256,
        "S41 authority contract digest",
    )

    executor_bundle = executor_bundle.resolve()
    validate_executor_bundle(
        executor_bundle,
        executor_bundle / "MANIFEST.json",
        sha256_text(
            executor_bundle_manifest_sha256,
            "S41 executor bundle manifest SHA-256",
        ),
    )
    contract_path = executor_bundle / CONTRACT_RELATIVE
    require(
        contract_path.is_file()
        and not contract_path.is_symlink()
        and contract_path.resolve() == contract_path,
        "S41 contract path",
    )
    raw = contract_path.read_bytes()
    require(
        0 < len(raw) <= MAX_COMMAND_BYTES,
        "S41 contract size",
    )
    require(
        hashlib.sha256(raw).hexdigest() == CONTRACT_SHA256,
        "S41 contract changed",
    )
    contract = strict_json_loads(raw, "S41 desktop smoke contract")
    require(canonical_bytes(contract) == raw, "S41 contract canonical bytes")
    contract = validate_contract_value(contract)

    require(expected_model_id in EXPECTED_PAIR, "S41 model outside pair")
    frozen_model = contract["models"][expected_model_id]
    require(
        value["gpu_uuid"] == contract["gpu"]["uuid"],
        "S41 authority GPU",
    )
    require(
        value["model_sha256"] == frozen_model["sha256"],
        "S41 authority model digest",
    )
    path = Path(expected_model_path)
    require(
        path.is_absolute()
        and path.is_file()
        and not path.is_symlink()
        and path.resolve() == path
        and value["model_path"] == expected_model_path,
        "S41 authority model path",
    )
    require(
        integer(value["model_bytes"], "S41 authority model bytes", 1)
        == frozen_model["bytes"]
        == path.stat().st_size,
        "S41 authority model size",
    )
    require(
        file_sha256(path) == frozen_model["sha256"],
        "S41 authority model changed",
    )
    validate_executor_bundle(
        executor_bundle,
        executor_bundle / "MANIFEST.json",
        executor_bundle_manifest_sha256,
    )
    return {
        "contract_sha256": CONTRACT_SHA256,
        "model_id": expected_model_id,
        "phase": "DESKTOP_SMOKE",
        "phase_id": phase_id,
        "schema": "s41-desktop-smoke-derived-v1",
        "scope": AUTHORITY_SCOPE,
        "status": "DESKTOP_B1_B8_SMOKE_AUTHORIZED",
    }
