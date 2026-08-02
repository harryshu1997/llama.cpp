#!/usr/bin/env python3
"""Independently validate the versioned S41 workload inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
GEMMA = "gemma-4-12b-it-q8_0"
QWEN = "qwen3-14b-q4_k_m"
MODEL_IDS = {GEMMA, QWEN}
MODEL_REMAP = {"qwen3-8b-q8_0": GEMMA, QWEN: QWEN}
REQUEST_SCHEMA = "s41-gemma-qwen-request-v1"
SWITCH_SCHEMA = "s41-gemma-qwen-switch-v1"

EXPECTED_MODELS = {
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
EXPECTED_SOURCES = {
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
REQUEST_KEYS = {
    "arrival_us",
    "event_id",
    "input_tokens",
    "model_id",
    "output_tokens",
    "prompt_tokens",
    "request_index",
    "schema",
    "slo_us",
    "source_input_tokens",
    "source_model",
    "source_output_tokens",
    "source_t_us",
}
SWITCH_KEYS = {
    "from_model_id",
    "intent_index",
    "kind",
    "schema",
    "source_event_id",
    "source_intent_id",
    "source_t_us",
    "t_us",
    "to_model_id",
}


class ValidationError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n").encode("ascii")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(data: bytes, label: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in values:
            if key in output:
                raise ValidationError(f"{label}: duplicate key {key}")
            output[key] = value
        return output

    try:
        return json.loads(
            data.decode("ascii"),
            object_pairs_hook=pairs,
            parse_float=lambda value: (_ for _ in ()).throw(
                ValidationError(f"{label}: float forbidden")),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValidationError(f"{label}: constant forbidden")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label}: invalid JSON") from exc


def read_canonical_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    data = path.read_bytes()
    value = strict_json(data, label)
    if type(value) is not dict or canonical(value) != data:
        raise ValidationError(f"{label}: non-canonical JSON")
    return data, value


def read_canonical_jsonl(
        path: Path, label: str) -> tuple[bytes, list[dict[str, Any]]]:
    data = path.read_bytes()
    if not data.endswith(b"\n") or b"\r" in data:
        raise ValidationError(f"{label}: non-canonical newline")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(data.splitlines()):
        value = strict_json(line + b"\n", f"{label}:{index + 1}")
        if type(value) is not dict or canonical(value) != line + b"\n":
            raise ValidationError(f"{label}:{index + 1}: non-canonical row")
        rows.append(value)
    return data, rows


def resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file():
        raise ValidationError(f"missing bound file: {relative}")
    return path


def validate_source_artifacts(
        root: Path, declared: Any) -> dict[str, bytes]:
    if declared != EXPECTED_SOURCES:
        raise ValidationError("source artifact binding changed")
    blobs: dict[str, bytes] = {}
    for relative, expected in EXPECTED_SOURCES.items():
        path = resolve(root, relative)
        data = path.read_bytes()
        if len(data) != expected["bytes"] or sha256(data) != expected["sha256"]:
            raise ValidationError(f"source artifact mismatch: {relative}")
        blobs[relative] = data
    return blobs


def expected_request(source: dict[str, Any]) -> dict[str, Any]:
    source_model = source.get("model_id")
    if source_model not in MODEL_REMAP:
        raise ValidationError("source request model invalid")
    output = dict(source)
    output["model_id"] = MODEL_REMAP[source_model]
    output["schema"] = REQUEST_SCHEMA
    return output


def expected_switch(source: dict[str, Any]) -> dict[str, Any]:
    from_model = source.get("from_model_id")
    to_model = source.get("to_model_id")
    if from_model not in MODEL_REMAP or to_model not in MODEL_REMAP:
        raise ValidationError("source switch model invalid")
    output = dict(source)
    output["from_model_id"] = MODEL_REMAP[from_model]
    output["to_model_id"] = MODEL_REMAP[to_model]
    output["schema"] = SWITCH_SCHEMA
    return output


def validate(root: Path = HERE) -> dict[str, Any]:
    manifest_bytes, manifest = read_canonical_json(
        root / "INPUT_MANIFEST.json", "manifest")
    if set(manifest) != {"builder_sha256", "files", "schema", "status"}:
        raise ValidationError("manifest key set changed")
    if manifest["schema"] != "s41-gemma-qwen-input-manifest-v1" \
            or manifest["status"] != "INPUTS_FROZEN_BEFORE_S41_ACQUISITION":
        raise ValidationError("manifest identity changed")
    builder_path = root / "build_inputs.py"
    if builder_path.is_file() \
            and sha256(builder_path.read_bytes()) != manifest["builder_sha256"]:
        raise ValidationError("builder digest mismatch")

    files = manifest.get("files")
    expected_file_names = {
        "REQUESTS.jsonl", "SWITCHES.jsonl", "INPUT_CONTRACT.json"}
    if type(files) is not dict or set(files) != expected_file_names:
        raise ValidationError("manifest file set changed")
    blobs: dict[str, bytes] = {}
    for name, record in files.items():
        if type(record) is not dict \
                or set(record) != {"bytes", "records", "sha256"}:
            raise ValidationError(f"{name}: manifest record invalid")
        data = (root / name).read_bytes()
        blobs[name] = data
        if record["bytes"] != len(data) or record["sha256"] != sha256(data):
            raise ValidationError(f"{name}: manifest binding failed")
        expected_records = 74 if name == "REQUESTS.jsonl" \
            else 9 if name == "SWITCHES.jsonl" else 1
        if record["records"] != expected_records:
            raise ValidationError(f"{name}: manifest records changed")

    contract_data = blobs["INPUT_CONTRACT.json"]
    contract = strict_json(contract_data, "contract")
    if type(contract) is not dict or canonical(contract) != contract_data:
        raise ValidationError("contract is not canonical")
    if set(contract) != {"models", "schema", "source", "status", "workload"}:
        raise ValidationError("contract key set changed")
    if contract["schema"] != "s41-gemma-qwen-input-contract-v1" \
            or contract["status"] != "INPUTS_FROZEN_BEFORE_S41_ACQUISITION":
        raise ValidationError("contract identity changed")
    if contract["models"] != EXPECTED_MODELS:
        raise ValidationError("model artifact binding changed")

    source = contract.get("source")
    if type(source) is not dict or set(source) != {
            "artifacts", "model_remap", "request_source", "switch_source",
            "transformation"}:
        raise ValidationError("source contract changed")
    if source["model_remap"] != MODEL_REMAP:
        raise ValidationError("model remap changed")
    if source["request_source"] != \
            "../s39_desktop_swap_baseline/DESKTOP_REQUESTS.jsonl" \
            or source["switch_source"] != \
            "../s39_desktop_swap_baseline/DESKTOP_SWITCHES.jsonl":
        raise ValidationError("direct source paths changed")
    expected_transformation = {
        "preserved_request_fields": sorted(REQUEST_KEYS - {"model_id", "schema"}),
        "preserved_switch_fields": sorted(
            SWITCH_KEYS - {"from_model_id", "schema", "to_model_id"}),
        "request_schema": REQUEST_SCHEMA,
        "switch_schema": SWITCH_SCHEMA,
    }
    if source["transformation"] != expected_transformation:
        raise ValidationError("source transformation changed")

    source_blobs = validate_source_artifacts(root, source["artifacts"])
    source_request_data = source_blobs[source["request_source"]]
    source_switch_data = source_blobs[source["switch_source"]]
    source_requests = [
        strict_json(line + b"\n", f"source request {index}")
        for index, line in enumerate(source_request_data.splitlines())
    ]
    source_switches = [
        strict_json(line + b"\n", f"source switch {index}")
        for index, line in enumerate(source_switch_data.splitlines())
    ]

    workload = contract.get("workload")
    if workload != {
        "initial_model_id": GEMMA,
        "output_tokens_per_request": 8,
        "prompt_semantics": "SYNTHETIC_GEOMETRY_ONLY_NOT_TASK_QUALITY",
        "request_count": 74,
        "requests_path": "REQUESTS.jsonl",
        "requests_sha256": sha256(blobs["REQUESTS.jsonl"]),
        "shared_token_id_upper_bound_exclusive": 151_936,
        "slo_us": 30_000_000,
        "switch_count": 9,
        "switches_path": "SWITCHES.jsonl",
        "switches_sha256": sha256(blobs["SWITCHES.jsonl"]),
        "token_requirement": "VALID_ID_IN_BOTH_BOUND_VOCABULARIES",
    }:
        raise ValidationError("workload contract changed")

    request_data, requests = read_canonical_jsonl(
        root / "REQUESTS.jsonl", "requests")
    switch_data, switches = read_canonical_jsonl(
        root / "SWITCHES.jsonl", "switches")
    if request_data != blobs["REQUESTS.jsonl"] \
            or switch_data != blobs["SWITCHES.jsonl"]:
        raise ValidationError("input changed between reads")
    if len(requests) != 74 or len(source_requests) != 74:
        raise ValidationError("request count changed")
    if len(switches) != 9 or len(source_switches) != 9:
        raise ValidationError("switch count changed")

    counts = {model_id: 0 for model_id in MODEL_IDS}
    common_vocab = min(
        model["vocabulary_size"] for model in EXPECTED_MODELS.values())
    maximum_token = -1
    seen: set[str] = set()
    for index, (row, source_row) in enumerate(zip(requests, source_requests)):
        if set(row) != REQUEST_KEYS or row != expected_request(source_row):
            raise ValidationError(f"request {index}: source geometry changed")
        if row["request_index"] != index:
            raise ValidationError(f"request {index}: index changed")
        event_id = row["event_id"]
        if type(event_id) is not str or not event_id or event_id in seen:
            raise ValidationError(f"request {index}: event identity invalid")
        seen.add(event_id)
        model_id = row["model_id"]
        if model_id not in counts:
            raise ValidationError(f"request {index}: model invalid")
        counts[model_id] += 1
        tokens = row["prompt_tokens"]
        if type(tokens) is not list or len(tokens) != row["input_tokens"]:
            raise ValidationError(f"request {index}: prompt length changed")
        for token in tokens:
            if type(token) is not int or token < 0 or token >= common_vocab:
                raise ValidationError(
                    f"request {index}: token invalid for common vocabulary")
            maximum_token = max(maximum_token, token)
        if row["output_tokens"] != 8 or row["slo_us"] != 30_000_000:
            raise ValidationError(f"request {index}: output or SLO changed")
    if counts != {GEMMA: 57, QWEN: 17}:
        raise ValidationError("request model mix changed")

    current = GEMMA
    request_by_event = {row["event_id"]: row for row in requests}
    previous_t = -1
    for index, (row, source_row) in enumerate(zip(switches, source_switches)):
        if set(row) != SWITCH_KEYS or row != expected_switch(source_row):
            raise ValidationError(f"switch {index}: source geometry changed")
        if row["intent_index"] != index \
                or row["from_model_id"] != current \
                or row["to_model_id"] not in MODEL_IDS \
                or row["to_model_id"] == current:
            raise ValidationError(f"switch {index}: route chain changed")
        if type(row["t_us"]) is not int or row["t_us"] <= previous_t:
            raise ValidationError(f"switch {index}: time changed")
        trigger = request_by_event.get(row["source_event_id"])
        if trigger is None or trigger["arrival_us"] != row["t_us"] \
                or trigger["model_id"] != row["to_model_id"]:
            raise ValidationError(f"switch {index}: trigger changed")
        current = row["to_model_id"]
        previous_t = row["t_us"]

    return {
        "contract_sha256": sha256(contract_data),
        "manifest_sha256": sha256(manifest_bytes),
        "maximum_token_id": maximum_token,
        "model_counts": counts,
        "request_count": len(requests),
        "status": "S41_INPUTS_VALID",
        "switch_count": len(switches),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"S41_INPUT_ERROR: {exc}")
        raise SystemExit(2)
