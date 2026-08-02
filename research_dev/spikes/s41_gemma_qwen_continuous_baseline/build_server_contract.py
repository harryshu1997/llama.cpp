#!/usr/bin/env python3
"""Build the S41 server-only acquisition contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import validate_inputs


HERE = Path(__file__).resolve().parent
INPUT_CONTRACT = HERE / "INPUT_CONTRACT.json"
INPUT_MANIFEST = HERE / "INPUT_MANIFEST.json"
OUTPUT = HERE / "SERVER_BASELINE_CONTRACT.json"
MANIFEST = HERE / "SERVER_BASELINE_MANIFEST.json"

GEMMA = "gemma-4-12b-it-q8_0"
QWEN = "qwen3-14b-q4_k_m"


def canonical(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n").encode("ascii")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    inputs = validate_inputs.validate(HERE)
    source = json.loads(INPUT_CONTRACT.read_text(encoding="ascii"))
    models = source["models"]
    contract = {
        "cache_regimes": {
            "COLD_NVME": {
                "maximum_resident_ppm": 50_000,
                "method": "POSIX_FADV_DONTNEED_COMPLETE_FILE",
            },
            "WARM_CACHE": {
                "minimum_resident_ppm": 950_000,
                "method": "COMPLETE_SEQUENTIAL_READ_BEFORE_PAID_TRACE",
            },
        },
        "device": {
            "gpu_memory_total_bytes": 17_175_674_880,
            "gpu_name": "NVIDIA GeForce RTX 4060 Ti",
            "gpu_uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            "host": "zhihao-Z690-C-ac",
            "ssh_target": "zhihao@172.20.74.85",
        },
        "energy_scope": {
            "measured": "SELECTED_GPU_BOARD",
            "server_wall": "UNKNOWN_NO_INSTRUMENT",
            "total_system": "UNKNOWN",
        },
        "inputs": {
            "contract_sha256": digest_file(INPUT_CONTRACT),
            "manifest_sha256": digest_file(INPUT_MANIFEST),
            "requests_path": "REQUESTS.jsonl",
            "requests_sha256": digest_file(HERE / "REQUESTS.jsonl"),
            "switches_path": "SWITCHES.jsonl",
            "switches_sha256": digest_file(HERE / "SWITCHES.jsonl"),
        },
        "models": {
            GEMMA: {
                **models[GEMMA],
                "path":
                    "/home/zhihao/models/gemma-4-12B-it-Q8_0-7b56.gguf",
            },
            QWEN: {
                **models[QWEN],
                "path": "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf",
            },
        },
        "non_coresidency": {
            "minimum_free_vram_bytes": 536_870_912,
            "requires_first_server_remain_healthy": True,
            "requires_physical_simultaneous_load_attempt": True,
        },
        "replay": {
            "cache_prompt": False,
            "ignore_eos": True,
            "initial_model_id": GEMMA,
            "output_tokens": 8,
            "request_count": 74,
            "run_order": [
                "WARM_CACHE",
                "COLD_NVME",
                "COLD_NVME",
                "WARM_CACHE",
                "WARM_CACHE",
                "COLD_NVME",
            ],
            "sampler": "greedy",
            "slo_us": 30_000_000,
            "switch_count": 9,
        },
        "schema": "s41-server-baseline-contract-v1",
        "serving": {
            "batch_size": 2048,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "context_size": 4096,
            "continuous_batching": True,
            "flash_attention": True,
            "gpu_layers": "all",
            "maximum_active_requests": 8,
            "minimum_free_vram_bytes": 536_870_912,
            "parallel_slots": 8,
            "physical_ubatch_size": 512,
            "split_mode": "none",
        },
        "status": "FROZEN_BEFORE_S41_SERVER_BASELINE_ACQUISITION",
    }
    payload = canonical(contract)
    OUTPUT.write_bytes(payload)
    manifest = {
        "files": {
            INPUT_CONTRACT.name: digest_file(INPUT_CONTRACT),
            INPUT_MANIFEST.name: digest_file(INPUT_MANIFEST),
            OUTPUT.name: hashlib.sha256(payload).hexdigest(),
            "REQUESTS.jsonl": digest_file(HERE / "REQUESTS.jsonl"),
            "SWITCHES.jsonl": digest_file(HERE / "SWITCHES.jsonl"),
        },
        "schema": "s41-server-baseline-manifest-v1",
        "status": "SERVER_CONTRACT_FROZEN_ACQUISITION_NOT_RUN",
    }
    MANIFEST.write_bytes(canonical(manifest))
    print(digest_file(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
