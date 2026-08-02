#!/usr/bin/env python3

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema


BUILDER_VERSION = "s38-distributed-rag-trace-v1"
EXPECTED_SOURCES = {
    "multihop_queries": {
        "revision": "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82",
        "license": "ODC-BY-1.0",
        "sha256": "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
        "records": 2556,
    },
    "multihop_corpus": {
        "revision": "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82",
        "license": "ODC-BY-1.0",
        "sha256": "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f",
        "records": 609,
    },
    "ragpulse": {
        "revision": "99a62769a91d5ebd17a2d4ddbbc88c1d16edc0e8",
        "license": "MIT",
        "sha256": "cd371571bef3320907147f8901729e37f412aafb067ec8eb153e2828e1801e65",
        "records": 7106,
    },
}
EXPECTED_OUTPUT_KEYS = {
    "op12_documents",
    "op15_documents",
    "payloads",
    "requests",
    "dense_requests",
}
EXPECTED_OUTPUT_PATHS = {
    "op12_documents": "op12_documents.jsonl",
    "op15_documents": "op15_documents.jsonl",
    "payloads": "payloads.jsonl",
    "requests": "requests.jsonl",
}


class ValidationError(RuntimeError):
    pass


def typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(typed_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_bytes(value: Any) -> bytes:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return (encoded + "\n").encode("ascii")


def file_sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid ASCII JSON in {path}: {error}") from None
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must contain one JSON object")
    if raw != canonical_bytes(value):
        raise ValidationError(f"{path} is not canonical JSON")
    return value


def read_schema(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid JSON schema in {path}: {error}") from None
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must contain one schema object")
    return value


def read_jsonl(path: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if type(record.get("bytes")) is not int or type(record.get("records")) is not int:
        raise ValidationError(f"invalid count type for {path.name}")
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValidationError(f"invalid SHA-256 field for {path.name}")
    if len(raw) != record.get("bytes"):
        raise ValidationError(f"byte count mismatch for {path.name}")
    if file_sha256_bytes(raw) != record.get("sha256"):
        raise ValidationError(f"SHA-256 mismatch for {path.name}")
    lines = raw.split(b"\n")
    if not lines or lines[-1] != b"" or any(not line for line in lines[:-1]):
        raise ValidationError(f"{path.name} must contain nonempty newline-terminated records")

    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines[:-1], 1):
        try:
            value = json.loads(line.decode("ascii"), object_pairs_hook=reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValidationError(f"invalid record {path.name}:{line_number}: {error}") from None
        if not isinstance(value, dict):
            raise ValidationError(f"{path.name}:{line_number} is not an object")
        if line + b"\n" != canonical_bytes(value):
            raise ValidationError(f"{path.name}:{line_number} is not canonical JSON")
        values.append(value)
    if len(values) != record.get("records"):
        raise ValidationError(f"record count mismatch for {path.name}")
    return values


def validate_documents(
    op12: list[dict[str, Any]],
    op15: list[dict[str, Any]],
) -> dict[str, str]:
    shard_by_id: dict[str, str] = {}
    for expected_shard, documents in (("op12", op12), ("op15", op15)):
        previous_id = ""
        for document in documents:
            if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
                raise ValidationError("document schema version must be integer 1")
            url = document.get("url")
            if not isinstance(url, str) or not url:
                raise ValidationError("document URL must be a nonempty string")
            document_id = "doc:" + hashlib.sha256(url.encode("utf-8")).hexdigest()
            if document.get("document_id") != document_id:
                raise ValidationError(f"document ID mismatch for {url}")
            shard = ("op12", "op15")[int(document_id[4:], 16) % 2]
            if shard != expected_shard or document.get("shard") != expected_shard:
                raise ValidationError(f"document shard mismatch for {document_id}")
            if document_id <= previous_id:
                raise ValidationError(f"document shard {expected_shard} is not strictly sorted")
            if document_id in shard_by_id:
                raise ValidationError(f"duplicate document ID: {document_id}")
            shard_by_id[document_id] = expected_shard
            previous_id = document_id
    if len(shard_by_id) != 609:
        raise ValidationError("document shards do not contain exactly 609 unique documents")
    return shard_by_id


def validate_payloads(
    payloads: list[dict[str, Any]],
    shard_by_id: dict[str, str],
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    query_indexes: set[int] = set()
    for payload in payloads:
        payload_id = payload.get("payload_id")
        query_index = payload.get("query_index")
        if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
            raise ValidationError("payload schema version must be integer 1")
        if not isinstance(payload_id, str) or payload_id in by_id:
            raise ValidationError("payload IDs must be unique strings")
        if type(query_index) is not int or query_index in query_indexes:
            raise ValidationError("query indexes must be unique integers")
        evidence_ids = payload.get("evidence_document_ids")
        if not isinstance(evidence_ids, list) or any(item not in shard_by_id for item in evidence_ids):
            raise ValidationError(f"payload {payload_id} has unresolved evidence")
        expected_shards = sorted({shard_by_id[item] for item in evidence_ids})
        if payload.get("evidence_shards") != expected_shards:
            raise ValidationError(f"payload {payload_id} has incorrect evidence shards")
        by_id[payload_id] = payload
        query_indexes.add(query_index)
    if query_indexes != set(range(2556)):
        raise ValidationError("payload query indexes are not the exact range 0..2555")
    return by_id


def validate_requests(
    requests: list[dict[str, Any]],
    payload_by_id: dict[str, dict[str, Any]],
    schema: dict[str, Any],
) -> dict[str, int]:
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)
    event_positions: dict[str, int] = {}
    seen_payloads: set[str] = set()
    previous_time = -1
    for index, request in enumerate(requests):
        errors = sorted(validator.iter_errors(request), key=lambda item: tuple(str(part) for part in item.absolute_path))
        if errors:
            raise ValidationError(f"request {index} fails schema: {errors[0].message}")
        validate_request_types(request, f"request {index}")
        event_id = request["event_id"]
        if event_id in event_positions:
            raise ValidationError(f"duplicate request event ID: {event_id}")
        if request["t_us"] < previous_time:
            raise ValidationError("request timestamps are not nondecreasing")
        payload_id = request["source_fields"].get("payload_id")
        if payload_id not in payload_by_id or payload_id in seen_payloads:
            raise ValidationError(f"request {event_id} has an invalid payload binding")
        payload = payload_by_id[payload_id]
        if request["source_fields"].get("evidence_count") != len(payload["evidence_document_ids"]):
            raise ValidationError(f"request {event_id} has an incorrect evidence count")
        expected_shards = ",".join(payload["evidence_shards"])
        if request["source_fields"].get("evidence_shards") != expected_shards:
            raise ValidationError(f"request {event_id} has incorrect evidence shards")
        event_positions[event_id] = index
        seen_payloads.add(payload_id)
        previous_time = request["t_us"]
    if seen_payloads != set(payload_by_id):
        raise ValidationError("requests do not bind every payload exactly once")
    return event_positions


def validate_request_types(request: dict[str, Any], label: str) -> None:
    for key in ("schema_version", "t_us", "input_tokens", "output_tokens", "images", "audio_ms", "retrieved_chunks"):
        if type(request.get(key)) is not int:
            raise ValidationError(f"{label} field {key} must be an integer")
    source_fields = request.get("source_fields")
    if not isinstance(source_fields, dict):
        raise ValidationError(f"{label} source_fields must be an object")
    for key in ("ragpulse_source_line", "ragpulse_timestamp_s", "evidence_count"):
        if type(source_fields.get(key)) is not int:
            raise ValidationError(f"{label} source field {key} must be an integer")


def validate_request_schema(requests: list[dict[str, Any]], schema: dict[str, Any], label: str) -> None:
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)
    previous_time = -1
    for index, request in enumerate(requests):
        errors = sorted(validator.iter_errors(request), key=lambda item: tuple(str(part) for part in item.absolute_path))
        if errors:
            raise ValidationError(f"{label} request {index} fails schema: {errors[0].message}")
        validate_request_types(request, f"{label} request {index}")
        if request["t_us"] < previous_time:
            raise ValidationError(f"{label} request timestamps are not nondecreasing")
        previous_time = request["t_us"]


def validate_dense(
    dense: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    event_positions: dict[str, int],
    dense_meta: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    start = dense_meta.get("source_start_index")
    end = dense_meta.get("source_end_index")
    denominator = dense_meta.get("time_scale_den")
    if type(start) is not int or type(end) is not int or type(denominator) is not int or denominator < 1:
        raise ValidationError("dense transform metadata has invalid integer fields")
    if not 0 <= start <= end < len(requests):
        raise ValidationError("dense transform range is outside the full trace")
    if type(dense_meta.get("time_scale_num")) is not int or dense_meta["time_scale_num"] != 1:
        raise ValidationError("dense transform numerator must be integer 1")
    source_span = requests[end]["t_us"] - requests[start]["t_us"]
    if type(dense_meta.get("source_span_us")) is not int or dense_meta["source_span_us"] != source_span:
        raise ValidationError("dense source span does not match the full trace")

    dense_positions = [event_positions.get(item.get("event_id"), -1) for item in dense]
    if dense_positions != list(range(start, end + 1)):
        raise ValidationError("dense requests are not the declared contiguous source window")
    if len(dense) != end - start + 1:
        raise ValidationError("dense window size does not match its declared range")

    base_time = requests[start]["t_us"]
    for index, dense_request in enumerate(dense):
        full_request = requests[start + index]
        expected_time = (full_request["t_us"] - base_time) // denominator
        expected = dict(full_request)
        expected["t_us"] = expected_time
        if dense_request != expected:
            raise ValidationError(f"dense request {index} is not the declared transform")
    if type(dense_meta.get("scaled_span_us")) is not int or dense_meta["scaled_span_us"] != dense[-1]["t_us"]:
        raise ValidationError("dense scaled span does not match the dense trace")
    validate_request_schema(dense, schema, "dense")


def validate_statistics(
    manifest: dict[str, Any],
    op12: list[dict[str, Any]],
    op15: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    requests: list[dict[str, Any]],
) -> None:
    expected = {
        "documents": len(op12) + len(op15),
        "op12_documents": len(op12),
        "op15_documents": len(op15),
        "requests": len(requests),
        "question_types": dict(sorted(collections.Counter(
            item["question_type"] for item in payloads
        ).items())),
        "evidence_counts": {str(key): value for key, value in sorted(collections.Counter(
            len(item["evidence_document_ids"]) for item in payloads
        ).items())},
        "evidence_shard_patterns": dict(sorted(collections.Counter(
            ",".join(item["evidence_shards"]) if item["evidence_shards"] else "none"
            for item in payloads
        ).items())),
    }
    if not typed_equal(manifest.get("statistics"), expected):
        raise ValidationError("manifest statistics do not match output records")


def validate_bundle(bundle: Path, schema_path: Path) -> None:
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("builder_version") != BUILDER_VERSION:
        raise ValidationError("unexpected builder version")
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1:
        raise ValidationError("unexpected manifest schema version")
    if manifest.get("provenance") != "semi_synthetic":
        raise ValidationError("unexpected manifest schema or provenance")
    if not typed_equal(manifest.get("sources"), EXPECTED_SOURCES):
        raise ValidationError("source pins do not match the frozen S38 inputs")
    transforms = manifest.get("transforms")
    expected_transforms = {
        "arrival_order": "stable sort by (integer timestamp, source line)",
        "arrival_selection": "first 2556 sorted RAGPulse records",
        "query_order": "ascending SHA-256 of builder-version/query-index",
        "document_id": "SHA-256 of UTF-8 URL",
        "document_shard": "integer document digest modulo 2; 0=op12, 1=op15",
        "priority_and_deadline": "unset; no real labels exist",
    }
    if not isinstance(transforms, dict) or set(transforms) != set(expected_transforms) | {"dense_window"}:
        raise ValidationError("manifest transform set is not exact")
    for key, expected in expected_transforms.items():
        if transforms.get(key) != expected:
            raise ValidationError(f"unexpected transform declaration: {key}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != EXPECTED_OUTPUT_KEYS:
        raise ValidationError("manifest output set is not exact")

    records: dict[str, list[dict[str, Any]]] = {}
    for key, output in outputs.items():
        if not isinstance(output, dict) or not isinstance(output.get("path"), str):
            raise ValidationError(f"invalid output record: {key}")
        if key == "dense_requests":
            dense_meta = transforms["dense_window"]
            start = dense_meta.get("source_start_index")
            end = dense_meta.get("source_end_index")
            denominator = dense_meta.get("time_scale_den")
            if (
                type(start) is not int
                or type(end) is not int
                or type(denominator) is not int
                or start < 0
                or end < start
                or denominator < 1
            ):
                raise ValidationError("dense transform cannot determine its output path")
            expected_path = f"dense{end - start + 1}_x{denominator}.jsonl"
        else:
            expected_path = EXPECTED_OUTPUT_PATHS[key]
        if output["path"] != expected_path:
            raise ValidationError(f"unexpected output path for {key}")
        records[key] = read_jsonl(bundle / output["path"], output)

    schema = read_schema(schema_path)
    shard_by_id = validate_documents(records["op12_documents"], records["op15_documents"])
    payload_by_id = validate_payloads(records["payloads"], shard_by_id)
    event_positions = validate_requests(records["requests"], payload_by_id, schema)
    validate_dense(
        records["dense_requests"],
        records["requests"],
        event_positions,
        manifest["transforms"]["dense_window"],
        schema,
    )
    validate_statistics(
        manifest,
        records["op12_documents"],
        records["op15_documents"],
        records["payloads"],
        records["requests"],
    )


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    repo = script.parents[3]
    default_schema = repo / "research_dev/spikes/s8_operator_island_affinity/schemas/request.schema.json"
    parser = argparse.ArgumentParser(description="Validate a frozen S38 trace bundle")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--request-schema", type=Path, default=default_schema)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        arguments = parse_args()
        validate_bundle(arguments.bundle.resolve(), arguments.request_schema.resolve())
    except (OSError, KeyError, TypeError, ValidationError, jsonschema.SchemaError) as error:
        raise SystemExit(f"S38_VALIDATION_ERROR: {error}") from None
    print("S38_BUNDLE_VALID")
