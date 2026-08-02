#!/usr/bin/env python3
"""Build the frozen CP0-D desktop replay inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "s39_phone_model_switch_trace/bundle_frequent"
REQUEST_SOURCE = SOURCE / "requests.jsonl"
SWITCH_SOURCE = SOURCE / "replay_intents.jsonl"
ACTIVE_TRACE = HERE.parent / "s39_phone_model_switch_trace/ACTIVE_TRACE.json"
PLAN = HERE / "PLAN.md"

REQUESTS_OUT = HERE / "DESKTOP_REQUESTS.jsonl"
SWITCHES_OUT = HERE / "DESKTOP_SWITCHES.jsonl"
CONTRACT_OUT = HERE / "DESKTOP_BASELINE_CONTRACT.json"
MANIFEST_OUT = HERE / "INPUT_MANIFEST.json"

ARRIVAL_SCALE_DEN = 20
INPUT_CAP = 128
OUTPUT_TOKENS = 8
SLO_US = 30_000_000
INITIAL_MODEL = "qwen3-8b-q8_0"
RUN_ORDER = [
    "WARM_CACHE",
    "COLD_NVME",
    "COLD_NVME",
    "WARM_CACHE",
    "WARM_CACHE",
    "COLD_NVME",
]
TOKEN_PALETTE = [
    34532, 425, 10965, 465, 374, 458, 6364, 4531,
    641, 220, 17, 15, 21, 1154, 1519, 1030,
    264, 45250, 3476, 304, 1378, 12351, 42074, 29253,
    320, 42882, 1365, 84831, 549, 28649, 15802, 15102,
]
MODEL_BY_SOURCE = {
    "ChatGPT": "qwen3-8b-q8_0",
    "GPT-4": "qwen3-14b-q4_k_m",
}
MODEL_REMAP = {
    "gemma-4-12b-it-q4_0": "qwen3-8b-q8_0",
    "qwen3-14b-q4_k_m": "qwen3-14b-q4_k_m",
}


class InputError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n").encode("ascii")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_bytes().splitlines()):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InputError(f"{path}:{index + 1}: invalid JSON") from exc
        if type(value) is not dict:
            raise InputError(f"{path}:{index + 1}: expected object")
        rows.append(value)
    return rows


def prompt_tokens(event_id: str, count: int) -> list[int]:
    seed = hashlib.sha256(event_id.encode("ascii")).digest()
    offset = seed[0] % len(TOKEN_PALETTE)
    step = 1 + 2 * (seed[1] % (len(TOKEN_PALETTE) // 2))
    return [
        TOKEN_PALETTE[(offset + index * step) % len(TOKEN_PALETTE)]
        for index in range(count)
    ]


def build_requests() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_index, row in enumerate(load_jsonl(REQUEST_SOURCE)):
        event_id = row.get("event_id")
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        source_fields = row.get("source_fields")
        t_us = row.get("t_us")
        if type(event_id) is not str or not event_id or event_id in seen:
            raise InputError(f"source row {source_index}: invalid event_id")
        seen.add(event_id)
        if type(output_tokens) is not int or output_tokens < 0:
            raise InputError(f"{event_id}: invalid output_tokens")
        if type(t_us) is not int or t_us < 0:
            raise InputError(f"{event_id}: invalid t_us")
        if type(source_fields) is not dict:
            raise InputError(f"{event_id}: missing source_fields")
        if output_tokens == 0:
            continue
        if type(input_tokens) is not int or input_tokens <= 0:
            raise InputError(f"{event_id}: invalid input_tokens")
        source_model = source_fields.get("model")
        if source_model not in MODEL_BY_SOURCE:
            raise InputError(f"{event_id}: unsupported source model")
        n_input = min(input_tokens, INPUT_CAP)
        output.append({
            "arrival_us": t_us // ARRIVAL_SCALE_DEN,
            "event_id": event_id,
            "input_tokens": n_input,
            "model_id": MODEL_BY_SOURCE[source_model],
            "output_tokens": OUTPUT_TOKENS,
            "prompt_tokens": prompt_tokens(event_id, n_input),
            "request_index": len(output),
            "schema": "s39-cp0d-desktop-request-v1",
            "slo_us": SLO_US,
            "source_input_tokens": input_tokens,
            "source_model": source_model,
            "source_output_tokens": output_tokens,
            "source_t_us": t_us,
        })
    output.sort(key=lambda row: (row["arrival_us"], row["request_index"]))
    for index, row in enumerate(output):
        row["request_index"] = index
    return output


def build_switches() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    previous = INITIAL_MODEL
    for index, row in enumerate(load_jsonl(SWITCH_SOURCE)):
        source_from = row.get("from_model_id")
        source_to = row.get("to_model_id")
        if source_from not in MODEL_REMAP or source_to not in MODEL_REMAP:
            raise InputError(f"switch {index}: unsupported model")
        from_model = MODEL_REMAP[source_from]
        to_model = MODEL_REMAP[source_to]
        if from_model != previous or from_model == to_model:
            raise InputError(f"switch {index}: non-alternating route")
        t_us = row.get("t_us")
        if type(t_us) is not int or t_us < 0:
            raise InputError(f"switch {index}: invalid t_us")
        output.append({
            "from_model_id": from_model,
            "intent_index": index,
            "kind": "TARGET_CHANGE",
            "schema": "s39-cp0d-desktop-switch-v1",
            "source_event_id": row.get("source_event_id"),
            "source_intent_id": row.get("intent_id"),
            "source_t_us": t_us,
            "t_us": t_us // ARRIVAL_SCALE_DEN,
            "to_model_id": to_model,
        })
        previous = to_model
    return output


def encode_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical(row) for row in rows)


def build_contract(request_sha: str, switch_sha: str) -> dict[str, Any]:
    return {
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
        "models": {
            "qwen3-14b-q4_k_m": {
                "bytes": 9_001_752_960,
                "path": "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf",
                "sha256": (
                    "500a8806e85ee9c83f3ae084202955924"
                    "51379b4f8cf2d0f41c15dffeb6b81f0"
                ),
            },
            "qwen3-8b-q8_0": {
                "bytes": 8_709_518_112,
                "path": "/home/zhihao/models/Qwen3-8B-Q8_0.gguf",
                "repository": "Qwen/Qwen3-8B-GGUF",
                "revision": "6cfbfc7d8ab95bf485c79fcc40be60930d5b4c8c",
                "sha256": (
                    "408b955510e196121c1c375201744783"
                    "b5c9a43c7956d73fc78df54c66e883d6"
                ),
            },
        },
        "non_coresidency": {
            "minimum_free_vram_bytes": 536_870_912,
            "requires_physical_simultaneous_load_attempt": True,
            "requires_first_server_remain_healthy": True,
        },
        "replay": {
            "arrival_scale_den": ARRIVAL_SCALE_DEN,
            "cache_prompt": False,
            "ignore_eos": True,
            "initial_model_id": INITIAL_MODEL,
            "input_token_cap": INPUT_CAP,
            "output_tokens": OUTPUT_TOKENS,
            "request_count": 74,
            "request_sha256": request_sha,
            "run_order": RUN_ORDER,
            "sampler": "greedy",
            "slo_us": SLO_US,
            "switch_count": 9,
            "switch_sha256": switch_sha,
            "token_palette": TOKEN_PALETTE,
        },
        "schema": "s39-cp0d-desktop-baseline-contract-v1",
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
        "source": {
            "active_trace_sha256": digest_file(ACTIVE_TRACE),
            "plan_sha256": digest_file(PLAN),
            "requests_sha256": digest_file(REQUEST_SOURCE),
            "switches_sha256": digest_file(SWITCH_SOURCE),
        },
        "status": "FROZEN_BEFORE_QWEN3_8B_DESKTOP_ACQUISITION",
        "switch_policy": [
            "ENQUEUE_FIFO",
            "ADMIT_PUBLISHED_MODEL_UP_TO_EIGHT",
            "CLOSE_ADMISSION_AT_INTENT",
            "DRAIN_ACTIVE_WITHOUT_CANCELLATION",
            "STOP_AND_WAIT_OLD_SERVER",
            "PREPARE_TARGET_CACHE_REGIME",
            "LOAD_FIXED_ENVELOPE",
            "PUBLISH_AFTER_HEALTH_AND_MEMORY_GATES",
            "REOPEN_TARGET_FIFO",
        ],
    }


def main() -> int:
    requests = build_requests()
    switches = build_switches()
    if len(requests) != 74 or len(switches) != 9:
        raise InputError("unexpected request or switch count")
    request_bytes = encode_jsonl(requests)
    switch_bytes = encode_jsonl(switches)
    request_sha = digest_bytes(request_bytes)
    switch_sha = digest_bytes(switch_bytes)
    contract = build_contract(request_sha, switch_sha)
    contract_bytes = canonical(contract)

    REQUESTS_OUT.write_bytes(request_bytes)
    SWITCHES_OUT.write_bytes(switch_bytes)
    CONTRACT_OUT.write_bytes(contract_bytes)

    manifest = {
        "builder_sha256": digest_file(Path(__file__)),
        "files": {
            REQUESTS_OUT.name: {
                "bytes": len(request_bytes),
                "records": len(requests),
                "sha256": request_sha,
            },
            SWITCHES_OUT.name: {
                "bytes": len(switch_bytes),
                "records": len(switches),
                "sha256": switch_sha,
            },
            CONTRACT_OUT.name: {
                "bytes": len(contract_bytes),
                "records": 1,
                "sha256": digest_bytes(contract_bytes),
            },
        },
        "schema": "s39-cp0d-input-manifest-v1",
        "status": "INPUTS_FROZEN_ACQUISITION_NOT_RUN",
    }
    MANIFEST_OUT.write_bytes(canonical(manifest))
    print(digest_file(MANIFEST_OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
