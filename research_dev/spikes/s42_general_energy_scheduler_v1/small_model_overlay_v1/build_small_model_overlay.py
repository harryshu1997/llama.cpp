#!/usr/bin/env python3
"""Derive a focused Llama 1B overlay from the six-model mixed trace."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
DONOR_ROOT = HERE.parent / "mixed_model_trace_v1"
DONOR_TRACE = DONOR_ROOT / "REQUESTS_MIXED_114.jsonl"
DONOR_MANIFEST = DONOR_ROOT / "TRACE_MANIFEST.json"
SOURCE_TRACE = (
    REPO_ROOT
    / "research_dev/spikes/s41_gemma_qwen_continuous_baseline"
    / "tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1"
    / "REQUESTS_SEMANTIC_SOURCE.jsonl"
)
TRACE = HERE / "REQUESTS_BURSTGPT_LLAMA1B_84.jsonl"
MANIFEST = HERE / "TRACE_MANIFEST.json"

TRACE_SCHEMA = "s42-three-model-burstgpt-small-overlay-v1"
MANIFEST_SCHEMA = (
    "s42-three-model-burstgpt-small-overlay-manifest-v1"
)
SOURCE_SCHEMA = "s41-gemma-qwen-request-semantic-source-v1"
SOURCE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4c"
    "31a4a9e8b4f9c085b29a14f80250a0ff"
)
DONOR_SCHEMA = "s42-six-model-burstgpt-mixed-v1"
DONOR_SHA256 = (
    "0622e5fe39d0f6f8bed8e0602680e5ed"
    "35941eeb571e686f235fa740ef61fae3"
)
ORIGINAL_STREAM = "burstgpt-original"
OVERLAY_STREAM = "llama-3.2-1b-overlay"
OVERLAY_MODEL = "llama-3.2-1b-instruct-q4_0"
SELECTED_STREAMS = frozenset({ORIGINAL_STREAM, OVERLAY_STREAM})


class BuildError(ValueError):
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
        raise BuildError("value is not canonical ASCII JSON") from error


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_object(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BuildError(f"cannot parse object: {path}") from error
    if type(value) is not dict or canonical(value) != content:
        raise BuildError(f"object is not canonical: {path}")
    return value, content


def load_rows(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise BuildError(f"cannot read trace: {path}") from error
    rows = []
    for line_number, raw in enumerate(content.splitlines(keepends=True), 1):
        if not raw.endswith(b"\n"):
            raise BuildError(f"line {line_number}: framing mismatch")
        try:
            row = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise BuildError(f"line {line_number}: invalid JSON") from error
        if type(row) is not dict or canonical(row) != raw:
            raise BuildError(f"line {line_number}: non-canonical row")
        rows.append(row)
    return rows, content


def derive() -> tuple[bytes, bytes]:
    source_rows, source_content = load_rows(SOURCE_TRACE)
    if (
        digest_bytes(source_content) != SOURCE_SHA256
        or len(source_rows) != 74
        or any(row.get("schema") != SOURCE_SCHEMA for row in source_rows)
    ):
        raise BuildError("source trace identity mismatch")

    donor_rows, donor_content = load_rows(DONOR_TRACE)
    donor_manifest, donor_manifest_content = load_object(DONOR_MANIFEST)
    if (
        digest_bytes(donor_content) != DONOR_SHA256
        or donor_manifest.get("trace", {}).get("sha256") != DONOR_SHA256
        or any(row.get("schema") != DONOR_SCHEMA for row in donor_rows)
    ):
        raise BuildError("donor trace identity mismatch")

    rows = [
        dict(row)
        for row in donor_rows
        if row.get("trace_stream_id") in SELECTED_STREAMS
    ]
    rows.sort(key=lambda row: (row["arrival_us"], row["event_id"]))
    if len(rows) != 84:
        raise BuildError("derived trace record count mismatch")
    if len({row["event_id"] for row in rows}) != len(rows):
        raise BuildError("derived trace event collision")
    for index, row in enumerate(rows):
        row["mixed_request_index"] = index
        row["schema"] = TRACE_SCHEMA
        if len(row.get("prompt_tokens", [])) != row.get("input_tokens"):
            raise BuildError("prompt token conservation mismatch")

    stream_counts = Counter(row["trace_stream_id"] for row in rows)
    model_counts = Counter(row["execution_model_id"] for row in rows)
    if stream_counts != Counter({ORIGINAL_STREAM: 74, OVERLAY_STREAM: 10}):
        raise BuildError("derived stream counts mismatch")
    if model_counts.get(OVERLAY_MODEL) != 10 or len(model_counts) != 3:
        raise BuildError("derived model counts mismatch")

    inventory = donor_manifest.get("model_inventory")
    if type(inventory) is not dict:
        raise BuildError("donor model inventory is absent")
    selected_inventory = {
        model_id: inventory[model_id] for model_id in sorted(model_counts)
    }
    trace_content = b"".join(canonical(row) for row in rows)
    manifest = {
        "arrival_first_us": rows[0]["arrival_us"],
        "arrival_last_us": rows[-1]["arrival_us"],
        "arrival_span_us": rows[-1]["arrival_us"] - rows[0]["arrival_us"],
        "derivation": {
            "donor_manifest_sha256": digest_bytes(donor_manifest_content),
            "donor_trace_path": str(DONOR_TRACE.relative_to(REPO_ROOT)),
            "donor_trace_schema": DONOR_SCHEMA,
            "donor_trace_sha256": DONOR_SHA256,
            "selected_streams": sorted(SELECTED_STREAMS),
        },
        "model_inventory": selected_inventory,
        "models": dict(sorted(model_counts.items())),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "record_count": len(rows),
        "schema": MANIFEST_SCHEMA,
        "small_model_overlay": {
            "arrival_offset_us": 200_000,
            "artifact_bytes": selected_inventory[OVERLAY_MODEL][
                "artifact_bytes"
            ],
            "artifact_sha256": selected_inventory[OVERLAY_MODEL][
                "artifact_sha256"
            ],
            "model_id": OVERLAY_MODEL,
            "request_count": stream_counts[OVERLAY_STREAM],
            "stream_id": OVERLAY_STREAM,
        },
        "source": {
            "path": str(SOURCE_TRACE.relative_to(REPO_ROOT)),
            "record_count": len(source_rows),
            "schema": SOURCE_SCHEMA,
            "sha256": SOURCE_SHA256,
        },
        "streams": dict(sorted(stream_counts.items())),
        "text_input_tokens": sum(row["text_input_tokens"] for row in rows),
        "trace": {
            "path": str(TRACE.relative_to(REPO_ROOT)),
            "schema": TRACE_SCHEMA,
            "sha256": digest_bytes(trace_content),
        },
    }
    return trace_content, canonical(manifest)


def write_or_compare(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise BuildError(f"existing output differs: {path}")
        return
    path.write_bytes(content)


def main() -> int:
    trace_content, manifest_content = derive()
    write_or_compare(TRACE, trace_content)
    write_or_compare(MANIFEST, manifest_content)
    manifest = json.loads(manifest_content)
    print(json.dumps({
        "manifest": str(MANIFEST),
        "models": manifest["models"],
        "records": manifest["record_count"],
        "trace": str(TRACE),
        "trace_sha256": manifest["trace"]["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
