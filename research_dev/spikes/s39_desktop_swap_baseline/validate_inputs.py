#!/usr/bin/env python3
"""Independently validate the frozen CP0-D desktop input bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
MODEL_IDS = {"qwen3-8b-q8_0", "qwen3-14b-q4_k_m"}


class ValidationError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n").encode("ascii")


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
                ValidationError(f"{label}: float forbidden")
            ),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValidationError(f"{label}: constant forbidden")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label}: invalid canonical JSON") from exc


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    data = path.read_bytes()
    if not data.endswith(b"\n") or b"\r" in data:
        raise ValidationError(f"{path.name}: non-canonical newline")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(data.splitlines()):
        value = strict_json(line + b"\n", f"{path.name}:{index + 1}")
        if type(value) is not dict or canonical(value) != line + b"\n":
            raise ValidationError(f"{path.name}:{index + 1}: non-canonical row")
        rows.append(value)
    return data, rows


def check_exact_keys(row: dict[str, Any], keys: set[str], label: str) -> None:
    if set(row) != keys:
        raise ValidationError(f"{label}: wrong keys")


def validate(root: Path) -> dict[str, Any]:
    manifest_path = root / "INPUT_MANIFEST.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = strict_json(manifest_bytes, "manifest")
    if type(manifest) is not dict or canonical(manifest) != manifest_bytes:
        raise ValidationError("manifest is not canonical")
    if manifest.get("schema") != "s39-cp0d-input-manifest-v1":
        raise ValidationError("wrong manifest schema")
    files = manifest.get("files")
    if type(files) is not dict or set(files) != {
        "DESKTOP_REQUESTS.jsonl",
        "DESKTOP_SWITCHES.jsonl",
        "DESKTOP_BASELINE_CONTRACT.json",
    }:
        raise ValidationError("wrong manifest file set")

    blobs: dict[str, bytes] = {}
    for name, record in files.items():
        path = root / name
        data = path.read_bytes()
        blobs[name] = data
        if type(record) is not dict:
            raise ValidationError(f"{name}: invalid manifest record")
        if record.get("bytes") != len(data) or record.get("sha256") != sha(data):
            raise ValidationError(f"{name}: manifest binding failed")

    contract = strict_json(blobs["DESKTOP_BASELINE_CONTRACT.json"], "contract")
    if type(contract) is not dict or canonical(contract) != blobs[
            "DESKTOP_BASELINE_CONTRACT.json"]:
        raise ValidationError("contract is not canonical")
    if contract.get("schema") != "s39-cp0d-desktop-baseline-contract-v1":
        raise ValidationError("wrong contract schema")
    replay = contract.get("replay")
    serving = contract.get("serving")
    models = contract.get("models")
    if type(replay) is not dict or type(serving) is not dict or type(models) is not dict:
        raise ValidationError("contract sections missing")
    if replay.get("run_order") != [
        "WARM_CACHE", "COLD_NVME", "COLD_NVME",
        "WARM_CACHE", "WARM_CACHE", "COLD_NVME",
    ]:
        raise ValidationError("run order changed")
    if replay.get("request_count") != 74 or replay.get("switch_count") != 9:
        raise ValidationError("trace cardinality changed")
    if serving.get("parallel_slots") != 8 \
            or serving.get("maximum_active_requests") != 8 \
            or serving.get("continuous_batching") is not True:
        raise ValidationError("serving envelope changed")
    if set(models) != MODEL_IDS:
        raise ValidationError("model set changed")

    request_data, requests = read_jsonl(root / "DESKTOP_REQUESTS.jsonl")
    switch_data, switches = read_jsonl(root / "DESKTOP_SWITCHES.jsonl")
    if replay.get("request_sha256") != sha(request_data) \
            or replay.get("switch_sha256") != sha(switch_data):
        raise ValidationError("contract does not bind replay bytes")
    if len(requests) != 74 or len(switches) != 9:
        raise ValidationError("wrong row count")

    request_keys = {
        "arrival_us", "event_id", "input_tokens", "model_id",
        "output_tokens", "prompt_tokens", "request_index", "schema", "slo_us",
        "source_input_tokens", "source_model", "source_output_tokens", "source_t_us",
    }
    seen: set[str] = set()
    previous_order: tuple[int, int] | None = None
    counts = {model: 0 for model in MODEL_IDS}
    palette = replay.get("token_palette")
    if type(palette) is not list or len(palette) != 32 \
            or any(type(value) is not int or value < 0 for value in palette):
        raise ValidationError("invalid token palette")
    palette_set = set(palette)
    for index, row in enumerate(requests):
        if type(row) is not dict:
            raise ValidationError(f"request {index}: not an object")
        check_exact_keys(row, request_keys, f"request {index}")
        if row.get("schema") != "s39-cp0d-desktop-request-v1" \
                or row.get("request_index") != index:
            raise ValidationError(f"request {index}: identity failed")
        event_id = row.get("event_id")
        if type(event_id) is not str or not event_id or event_id in seen:
            raise ValidationError(f"request {index}: duplicate event")
        seen.add(event_id)
        model_id = row.get("model_id")
        if model_id not in MODEL_IDS:
            raise ValidationError(f"request {index}: model invalid")
        counts[model_id] += 1
        arrival = row.get("arrival_us")
        source_t = row.get("source_t_us")
        input_tokens = row.get("input_tokens")
        source_input = row.get("source_input_tokens")
        tokens = row.get("prompt_tokens")
        if type(arrival) is not int or type(source_t) is not int \
                or arrival != source_t // 20:
            raise ValidationError(f"request {index}: arrival mapping failed")
        if type(source_input) is not int or type(input_tokens) is not int \
                or input_tokens != min(source_input, 128):
            raise ValidationError(f"request {index}: input mapping failed")
        if row.get("output_tokens") != 8 or row.get("slo_us") != 30_000_000:
            raise ValidationError(f"request {index}: output or SLO changed")
        if type(tokens) is not list or len(tokens) != input_tokens \
                or any(type(value) is not int or value not in palette_set for value in tokens):
            raise ValidationError(f"request {index}: token payload invalid")
        order = (arrival, index)
        if previous_order is not None and order <= previous_order:
            raise ValidationError(f"request {index}: ordering failed")
        previous_order = order
    if counts != {"qwen3-8b-q8_0": 57, "qwen3-14b-q4_k_m": 17}:
        raise ValidationError("model request counts changed")

    switch_keys = {
        "from_model_id", "intent_index", "kind", "schema", "source_event_id",
        "source_intent_id", "source_t_us", "t_us", "to_model_id",
    }
    current = "qwen3-8b-q8_0"
    previous_t = -1
    for index, row in enumerate(switches):
        if type(row) is not dict:
            raise ValidationError(f"switch {index}: not an object")
        check_exact_keys(row, switch_keys, f"switch {index}")
        if row.get("schema") != "s39-cp0d-desktop-switch-v1" \
                or row.get("intent_index") != index \
                or row.get("kind") != "TARGET_CHANGE":
            raise ValidationError(f"switch {index}: identity failed")
        if row.get("from_model_id") != current \
                or row.get("to_model_id") not in MODEL_IDS \
                or row.get("to_model_id") == current:
            raise ValidationError(f"switch {index}: route failed")
        if type(row.get("source_t_us")) is not int \
                or row.get("t_us") != row["source_t_us"] // 20 \
                or row["t_us"] <= previous_t:
            raise ValidationError(f"switch {index}: timing failed")
        current = row["to_model_id"]
        previous_t = row["t_us"]

    return {
        "contract_sha256": sha(blobs["DESKTOP_BASELINE_CONTRACT.json"]),
        "manifest_sha256": sha(manifest_bytes),
        "model_counts": counts,
        "request_count": len(requests),
        "status": "CP0D_INPUTS_VALID",
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
        print(f"CP0D_INPUT_ERROR: {exc}")
        raise SystemExit(2)
