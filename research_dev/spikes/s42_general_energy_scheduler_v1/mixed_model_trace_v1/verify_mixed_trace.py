#!/usr/bin/env python3
"""Verify the six-model BurstGPT mixed trace and its source bindings."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
TRACE = HERE / "REQUESTS_MIXED_114.jsonl"
MANIFEST = HERE / "TRACE_MANIFEST.json"

TRACE_SCHEMA = "s42-six-model-burstgpt-mixed-v1"
MANIFEST_SCHEMA = "s42-six-model-burstgpt-mixed-manifest-v1"
SOURCE_SCHEMA = "s41-gemma-qwen-request-semantic-source-v1"
SOURCE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4c"
    "31a4a9e8b4f9c085b29a14f80250a0ff"
)
EXPECTED_STREAMS = {
    "burstgpt-original": (74, None, None),
    "gemma-4-e2b-vlm-overlay": (10, "gemma-4-e2b-it-q8_0-vlm", 400_000),
    "llama-3.2-1b-overlay": (10, "llama-3.2-1b-instruct-q4_0", 200_000),
    "qwen3-0.6b-overlay": (10, "qwen3-0.6b-q8_0", 100_000),
    "qwen3-8b-overlay": (10, "qwen3-8b-q8_0", 300_000),
}
EXPECTED_MODELS = {
    "gemma-4-12b-it-q4_0": 17,
    "gemma-4-e2b-it-q8_0-vlm": 10,
    "llama-3.2-1b-instruct-q4_0": 10,
    "qwen3-0.6b-q8_0": 10,
    "qwen3-14b-q4_k_m": 57,
    "qwen3-8b-q8_0": 10,
}


class TraceValidationError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError, UnicodeError) as error:
        raise TraceValidationError("value is not canonical ASCII JSON") from error


def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise TraceValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_one(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=no_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TraceValidationError(f"cannot parse {path}") from error
    if type(value) is not dict or canonical(value) != raw:
        raise TraceValidationError(f"non-canonical object: {path}")
    return value


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[bytes], bytes]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise TraceValidationError(f"cannot read {path}") from error
    rows = []
    raw_rows = []
    for line_number, raw in enumerate(content.splitlines(keepends=True), 1):
        if not raw.endswith(b"\n"):
            raise TraceValidationError(f"line {line_number}: framing mismatch")
        try:
            row = json.loads(raw, object_pairs_hook=no_duplicates)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise TraceValidationError(f"line {line_number}: invalid JSON") from error
        if type(row) is not dict or canonical(row) != raw:
            raise TraceValidationError(f"line {line_number}: non-canonical row")
        rows.append(row)
        raw_rows.append(raw)
    return rows, raw_rows, content


def strict_int(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TraceValidationError(f"{name} must be an integer >= {minimum}")
    return value


def strict_text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise TraceValidationError(f"{name} must be non-empty text")
    try:
        value.encode("ascii")
    except UnicodeError as error:
        raise TraceValidationError(f"{name} must be ASCII") from error
    return value


def validate_parent(
    row: dict[str, Any],
    parent: dict[str, Any],
    parent_raw: bytes,
) -> None:
    if row.get("parent_schema") != parent.get("schema"):
        raise TraceValidationError("parent schema mismatch")
    if row.get("parent_record_sha256") != digest_bytes(parent_raw):
        raise TraceValidationError("parent record hash mismatch")
    for key, value in parent.items():
        if key == "schema":
            continue
        if row.get(key) != value:
            raise TraceValidationError(f"parent field mismatch: {key}")
    if row.get("stream_request_index") != parent.get("request_index"):
        raise TraceValidationError("parent stream index mismatch")
    execution_model = parent.get("prompt_tokenizer_model")
    if row.get("requested_model_id") != execution_model:
        raise TraceValidationError("parent requested model mismatch")
    if row.get("execution_model_id") != execution_model:
        raise TraceValidationError("parent execution model mismatch")


def validate_overlay(
    row: dict[str, Any],
    parent_by_event: dict[str, dict[str, Any]],
    expected_model: str,
    expected_offset: int,
) -> None:
    donor_id = strict_text("source_shape_event_id", row.get("source_shape_event_id"))
    donor = parent_by_event.get(donor_id)
    if donor is None:
        raise TraceValidationError("overlay donor is absent")
    if row.get("source_shape_record_sha256") != digest_bytes(canonical(donor)):
        raise TraceValidationError("overlay donor hash mismatch")
    if row.get("arrival_us") != donor.get("arrival_us") + expected_offset:
        raise TraceValidationError("overlay arrival mismatch")
    if row.get("slo_us") != donor.get("slo_us"):
        raise TraceValidationError("overlay SLO mismatch")
    if row.get("requested_model_id") != expected_model:
        raise TraceValidationError("overlay requested model mismatch")
    if row.get("execution_model_id") != expected_model:
        raise TraceValidationError("overlay execution model mismatch")
    if row.get("model_id") != expected_model:
        raise TraceValidationError("overlay model mismatch")
    for key in ("source_input_tokens", "source_output_tokens", "source_t_us"):
        if row.get(key) != donor.get(key):
            raise TraceValidationError(f"overlay donor field mismatch: {key}")
    if row.get("modality") == "text":
        if row.get("input_tokens") != donor.get("input_tokens"):
            raise TraceValidationError("text overlay input geometry mismatch")
        if row.get("output_tokens") != donor.get("output_tokens"):
            raise TraceValidationError("text overlay output geometry mismatch")


def validate_quality_case(value: object) -> None:
    if type(value) is not dict:
        raise TraceValidationError("quality case must be an object")
    strict_text("quality case id", value.get("case_id"))
    strict_text("quality question", value.get("question"))
    if value.get("scorer") != "normalized_contains_each_group_v1":
        raise TraceValidationError("quality scorer mismatch")
    groups = value.get("answer_groups")
    if type(groups) is not list or not groups:
        raise TraceValidationError("quality answer groups are absent")
    for group in groups:
        if type(group) is not list or not group:
            raise TraceValidationError("quality answer group is empty")
        for answer in group:
            strict_text("quality answer", answer)


def validate_image_row(
    row: dict[str, Any],
    manifest: dict[str, Any],
    check_assets: bool,
) -> None:
    image = row.get("image")
    if type(image) is not dict:
        raise TraceValidationError("VLM image is absent")
    image_id = strict_text("image id", image.get("id"))
    inventory = manifest.get("image_assets")
    if type(inventory) is not dict or image_id not in inventory:
        raise TraceValidationError("VLM image is not in the inventory")
    expected = dict(inventory[image_id])
    expected["id"] = image_id
    if image != expected:
        raise TraceValidationError("VLM image metadata mismatch")
    image_bytes = strict_int("image bytes", image.get("bytes"), 1)
    width = strict_int("image width", image.get("width"), 1)
    height = strict_int("image height", image.get("height"), 1)
    if row.get("image_count") != 1:
        raise TraceValidationError("VLM image count mismatch")
    if row.get("image_bytes") != image_bytes:
        raise TraceValidationError("VLM image bytes mismatch")
    if row.get("image_pixels") != width * height:
        raise TraceValidationError("VLM image pixels mismatch")
    if row.get("image_token_count_status") != "runtime_measured":
        raise TraceValidationError("VLM image token status mismatch")
    if "image_tokens" in row:
        raise TraceValidationError("unmeasured image token count was populated")
    model = manifest["model_inventory"][row["execution_model_id"]]
    expected_cache_key = (
        f"sha256:{image['sha256']}:mmproj:{model['projector_sha256']}"
    )
    if row.get("image_cache_key") != expected_cache_key:
        raise TraceValidationError("VLM image cache key mismatch")
    if row.get("vision_projector_bytes") != model.get("projector_bytes"):
        raise TraceValidationError("VLM projector size mismatch")
    if row.get("vision_projector_sha256") != model.get("projector_sha256"):
        raise TraceValidationError("VLM projector hash mismatch")
    validate_quality_case(row.get("quality_case"))
    if row.get("prompt_transport") != "multimodal_message":
        raise TraceValidationError("VLM prompt transport mismatch")
    if row.get("prompt_text") != row["quality_case"].get("question"):
        raise TraceValidationError("VLM raw prompt mismatch")
    if check_assets:
        path = REPO_ROOT / image["path"]
        try:
            size = path.stat().st_size
        except OSError as error:
            raise TraceValidationError("VLM image asset is absent") from error
        if size != image_bytes or digest_file(path) != image["sha256"]:
            raise TraceValidationError("VLM image asset identity mismatch")


def validate(
    trace_path: Path = TRACE,
    manifest_path: Path = MANIFEST,
    check_assets: bool = True,
) -> dict[str, Any]:
    manifest = load_one(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise TraceValidationError("manifest schema mismatch")
    if manifest.get("record_count") != 114:
        raise TraceValidationError("manifest record count mismatch")
    source_meta = manifest.get("source")
    if type(source_meta) is not dict:
        raise TraceValidationError("source metadata is absent")
    if (
        source_meta.get("schema") != SOURCE_SCHEMA
        or source_meta.get("sha256") != SOURCE_SHA256
        or source_meta.get("record_count") != 74
    ):
        raise TraceValidationError("source metadata mismatch")
    source_path = REPO_ROOT / strict_text("source path", source_meta.get("path"))
    parent_rows, parent_raw_rows, source_content = load_rows(source_path)
    if digest_bytes(source_content) != SOURCE_SHA256 or len(parent_rows) != 74:
        raise TraceValidationError("source trace identity mismatch")
    parent_by_event = {
        strict_text("parent event", row.get("event_id")): row
        for row in parent_rows
    }
    raw_by_event = {
        parent_rows[index]["event_id"]: parent_raw_rows[index]
        for index in range(len(parent_rows))
    }

    rows, _, trace_content = load_rows(trace_path)
    trace_meta = manifest.get("trace")
    if type(trace_meta) is not dict:
        raise TraceValidationError("trace metadata is absent")
    if trace_meta.get("schema") != TRACE_SCHEMA:
        raise TraceValidationError("trace schema metadata mismatch")
    if trace_meta.get("sha256") != digest_bytes(trace_content):
        raise TraceValidationError("trace hash mismatch")
    if len(rows) != 114:
        raise TraceValidationError("trace record count mismatch")

    seen = set()
    previous = None
    for index, row in enumerate(rows):
        if row.get("schema") != TRACE_SCHEMA:
            raise TraceValidationError("row schema mismatch")
        if row.get("mixed_request_index") != index:
            raise TraceValidationError("mixed request index mismatch")
        event_id = strict_text("event id", row.get("event_id"))
        if event_id in seen:
            raise TraceValidationError("duplicate event id")
        seen.add(event_id)
        arrival_us = strict_int("arrival_us", row.get("arrival_us"))
        key = (arrival_us, event_id)
        if previous is not None and key < previous:
            raise TraceValidationError("trace is not in arrival order")
        previous = key
        input_tokens = strict_int("input_tokens", row.get("input_tokens"), 1)
        strict_int("output_tokens", row.get("output_tokens"), 1)
        strict_int("slo_us", row.get("slo_us"), 1)
        tokens = row.get("prompt_tokens")
        if (
            type(tokens) is not list
            or len(tokens) != input_tokens
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise TraceValidationError("prompt token conservation mismatch")
        if row.get("text_input_tokens") != input_tokens:
            raise TraceValidationError("text input token mismatch")
        stream_id = strict_text("trace stream id", row.get("trace_stream_id"))
        if stream_id not in EXPECTED_STREAMS:
            raise TraceValidationError("unknown trace stream")
        if stream_id != "burstgpt-original":
            strict_text("formatted prompt text", row.get("formatted_prompt_text"))
        expected_model = EXPECTED_STREAMS[stream_id][1]
        if stream_id == "burstgpt-original":
            parent = parent_by_event.get(event_id)
            if parent is None:
                raise TraceValidationError("original event is absent from source")
            validate_parent(row, parent, raw_by_event[event_id])
        else:
            assert expected_model is not None
            expected_offset = EXPECTED_STREAMS[stream_id][2]
            assert expected_offset is not None
            validate_overlay(row, parent_by_event, expected_model, expected_offset)
        model_id = strict_text("execution model", row.get("execution_model_id"))
        model_inventory = manifest.get("model_inventory")
        if type(model_inventory) is not dict or model_id not in model_inventory:
            raise TraceValidationError("execution model is not inventoried")
        model = model_inventory[model_id]
        if row.get("model_artifact_bytes") != model.get("artifact_bytes"):
            raise TraceValidationError("model artifact size mismatch")
        if row.get("model_artifact_sha256") != model.get("artifact_sha256"):
            raise TraceValidationError("model artifact hash mismatch")
        modality = row.get("modality")
        if modality == "image_text":
            validate_image_row(row, manifest, check_assets)
        elif modality != "text":
            raise TraceValidationError("unknown modality")
        elif row.get("prompt_transport") != "tokens":
            raise TraceValidationError("text prompt transport mismatch")

    stream_counts = Counter(row["trace_stream_id"] for row in rows)
    model_counts = Counter(row["execution_model_id"] for row in rows)
    expected_stream_counts = {
        stream: details[0] for stream, details in EXPECTED_STREAMS.items()
    }
    if dict(stream_counts) != expected_stream_counts:
        raise TraceValidationError("stream counts mismatch")
    if dict(model_counts) != EXPECTED_MODELS:
        raise TraceValidationError("model counts mismatch")
    if manifest.get("streams") != dict(sorted(stream_counts.items())):
        raise TraceValidationError("manifest stream counts mismatch")
    if manifest.get("models") != dict(sorted(model_counts.items())):
        raise TraceValidationError("manifest model counts mismatch")
    image_rows = [row for row in rows if row["modality"] == "image_text"]
    checks = {
        "arrival_first_us": rows[0]["arrival_us"],
        "arrival_last_us": rows[-1]["arrival_us"],
        "arrival_span_us": rows[-1]["arrival_us"] - rows[0]["arrival_us"],
        "image_request_count": len(image_rows),
        "image_unique_count": len({row["image"]["sha256"] for row in image_rows}),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "text_input_tokens": sum(row["text_input_tokens"] for row in rows),
    }
    for key, value in checks.items():
        if manifest.get(key) != value:
            raise TraceValidationError(f"manifest aggregate mismatch: {key}")
    if manifest.get("vlm_image_tokens") != {
        "status": "runtime_measured",
        "value": None,
    }:
        raise TraceValidationError("manifest VLM image token boundary mismatch")
    return {
        "models": dict(sorted(model_counts.items())),
        "output_tokens": checks["output_tokens"],
        "records": len(rows),
        "text_input_tokens": checks["text_input_tokens"],
        "trace_sha256": digest_bytes(trace_content),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--skip-asset-check", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(
            args.trace,
            args.manifest,
            check_assets=not args.skip_asset_check,
        )
    except TraceValidationError as error:
        raise SystemExit(f"FAIL: {error}") from error
    print(json.dumps({"verdict": "PASS", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
