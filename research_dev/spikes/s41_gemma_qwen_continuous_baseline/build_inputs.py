#!/usr/bin/env python3
"""Build the versioned S41 Gemma/Qwen workload inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
S39_DESKTOP = HERE.parent / "s39_desktop_swap_baseline"
S39_TRACE = HERE.parent / "s39_phone_model_switch_trace"

REQUEST_SOURCE = S39_DESKTOP / "DESKTOP_REQUESTS.jsonl"
SWITCH_SOURCE = S39_DESKTOP / "DESKTOP_SWITCHES.jsonl"

REQUESTS_OUT = HERE / "REQUESTS.jsonl"
SWITCHES_OUT = HERE / "SWITCHES.jsonl"
CONTRACT_OUT = HERE / "INPUT_CONTRACT.json"
MANIFEST_OUT = HERE / "INPUT_MANIFEST.json"

GEMMA = "gemma-4-12b-it-q8_0"
QWEN = "qwen3-14b-q4_k_m"
MODEL_REMAP = {
    "qwen3-8b-q8_0": GEMMA,
    QWEN: QWEN,
}
REQUEST_SCHEMA = "s41-gemma-qwen-request-v1"
SWITCH_SCHEMA = "s41-gemma-qwen-switch-v1"

SOURCE_ARTIFACTS = {
    "../s39_desktop_swap_baseline/DESKTOP_BASELINE_CONTRACT.json": {
        "bytes": 2768,
        "sha256":
            "b2e87de77c86d2ff880882dea8210c9cff314b2b63ff673c93836d9c86a77d2f",
    },
    "../s39_desktop_swap_baseline/DESKTOP_REQUESTS.jsonl": {
        "bytes": 64283,
        "sha256":
            "c6b99c54bf55ffc885057d074780eea99000bceace9eedb64c4ce7eb6c537145",
    },
    "../s39_desktop_swap_baseline/DESKTOP_SWITCHES.jsonl": {
        "bytes": 2950,
        "sha256":
            "965c20d9296dfcc27f7d9b7258c44aaec72ebde50486f320e30ef0eaa2d6dd4c",
    },
    "../s39_desktop_swap_baseline/INPUT_MANIFEST.json": {
        "bytes": 573,
        "sha256":
            "543a788a2cad9327c4b84886e0db8d6586cac05c30dcad7673824c5d927b6cc3",
    },
    "../s39_phone_model_switch_trace/ACTIVE_TRACE.json": {
        "bytes": 159,
        "sha256":
            "ca3d55617ecca995ef8e1ed2795776bbf703fe8ee5593592fc844c8a741afae8",
    },
    "../s39_phone_model_switch_trace/bundle_frequent/replay_intents.jsonl": {
        "bytes": 3452,
        "sha256":
            "df5f3e66efa77bf7ae9c722f84a059f109aa91fce908013b54003bc9d984ebb7",
    },
    "../s39_phone_model_switch_trace/bundle_frequent/requests.jsonl": {
        "bytes": 42926,
        "sha256":
            "f5ce938bd7062a92c36a1b21384cf5f8a926f7ed47aa52a7472a13d8d7628700",
    },
}

MODELS = {
    GEMMA: {
        "architecture": "gemma4",
        "bytes": 12_669_645_856,
        "quantization": "Q8_0",
        "sha256":
            "7b56cbd0e0d96d5c8d7df9c21b39d67264e58cbabf7a96d4681eff8b3d492848",
        "vocabulary_size": 262_144,
    },
    QWEN: {
        "architecture": "qwen3",
        "bytes": 9_001_752_960,
        "quantization": "Q4_K_M",
        "sha256":
            "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
        "vocabulary_size": 151_936,
    },
}


class InputError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n").encode("ascii")


def sha256(data: bytes) -> str:
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


def check_sources() -> None:
    for relative, expected in SOURCE_ARTIFACTS.items():
        path = (HERE / relative).resolve()
        if not path.is_file():
            raise InputError(f"missing source artifact: {relative}")
        if path.stat().st_size != expected["bytes"]:
            raise InputError(f"source artifact byte mismatch: {relative}")
        if digest_file(path) != expected["sha256"]:
            raise InputError(f"source artifact digest mismatch: {relative}")


def build_requests() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, source in enumerate(load_jsonl(REQUEST_SOURCE)):
        if source.get("request_index") != index:
            raise InputError(f"request {index}: source index mismatch")
        source_model = source.get("model_id")
        if source_model not in MODEL_REMAP:
            raise InputError(f"request {index}: unsupported source model")
        row = dict(source)
        row["model_id"] = MODEL_REMAP[source_model]
        row["schema"] = REQUEST_SCHEMA
        output.append(row)
    return output


def build_switches() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, source in enumerate(load_jsonl(SWITCH_SOURCE)):
        if source.get("intent_index") != index:
            raise InputError(f"switch {index}: source index mismatch")
        from_model = source.get("from_model_id")
        to_model = source.get("to_model_id")
        if from_model not in MODEL_REMAP or to_model not in MODEL_REMAP:
            raise InputError(f"switch {index}: unsupported source model")
        row = dict(source)
        row["from_model_id"] = MODEL_REMAP[from_model]
        row["to_model_id"] = MODEL_REMAP[to_model]
        row["schema"] = SWITCH_SCHEMA
        output.append(row)
    return output


def encode_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical(row) for row in rows)


def build_contract(request_sha: str, switch_sha: str) -> dict[str, Any]:
    return {
        "models": MODELS,
        "schema": "s41-gemma-qwen-input-contract-v1",
        "source": {
            "artifacts": SOURCE_ARTIFACTS,
            "model_remap": MODEL_REMAP,
            "request_source":
                "../s39_desktop_swap_baseline/DESKTOP_REQUESTS.jsonl",
            "switch_source":
                "../s39_desktop_swap_baseline/DESKTOP_SWITCHES.jsonl",
            "transformation": {
                "preserved_request_fields": [
                    "arrival_us",
                    "event_id",
                    "input_tokens",
                    "output_tokens",
                    "prompt_tokens",
                    "request_index",
                    "slo_us",
                    "source_input_tokens",
                    "source_model",
                    "source_output_tokens",
                    "source_t_us",
                ],
                "preserved_switch_fields": [
                    "intent_index",
                    "kind",
                    "source_event_id",
                    "source_intent_id",
                    "source_t_us",
                    "t_us",
                ],
                "request_schema": REQUEST_SCHEMA,
                "switch_schema": SWITCH_SCHEMA,
            },
        },
        "status": "INPUTS_FROZEN_BEFORE_S41_ACQUISITION",
        "workload": {
            "initial_model_id": GEMMA,
            "output_tokens_per_request": 8,
            "prompt_semantics": "SYNTHETIC_GEOMETRY_ONLY_NOT_TASK_QUALITY",
            "request_count": 74,
            "requests_path": "REQUESTS.jsonl",
            "requests_sha256": request_sha,
            "shared_token_id_upper_bound_exclusive": 151_936,
            "slo_us": 30_000_000,
            "switch_count": 9,
            "switches_path": "SWITCHES.jsonl",
            "switches_sha256": switch_sha,
            "token_requirement": "VALID_ID_IN_BOTH_BOUND_VOCABULARIES",
        },
    }


def main() -> int:
    check_sources()
    requests = build_requests()
    switches = build_switches()
    if len(requests) != 74 or len(switches) != 9:
        raise InputError("unexpected request or switch count")

    request_bytes = encode_jsonl(requests)
    switch_bytes = encode_jsonl(switches)
    contract_bytes = canonical(build_contract(
        sha256(request_bytes), sha256(switch_bytes)))

    REQUESTS_OUT.write_bytes(request_bytes)
    SWITCHES_OUT.write_bytes(switch_bytes)
    CONTRACT_OUT.write_bytes(contract_bytes)

    files = {
        REQUESTS_OUT.name: {
            "bytes": len(request_bytes),
            "records": len(requests),
            "sha256": sha256(request_bytes),
        },
        SWITCHES_OUT.name: {
            "bytes": len(switch_bytes),
            "records": len(switches),
            "sha256": sha256(switch_bytes),
        },
        CONTRACT_OUT.name: {
            "bytes": len(contract_bytes),
            "records": 1,
            "sha256": sha256(contract_bytes),
        },
    }
    manifest = {
        "builder_sha256": digest_file(Path(__file__)),
        "files": files,
        "schema": "s41-gemma-qwen-input-manifest-v1",
        "status": "INPUTS_FROZEN_BEFORE_S41_ACQUISITION",
    }
    MANIFEST_OUT.write_bytes(canonical(manifest))
    print(digest_file(MANIFEST_OUT))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputError as exc:
        print(f"S41_INPUT_BUILD_ERROR: {exc}")
        raise SystemExit(2)
