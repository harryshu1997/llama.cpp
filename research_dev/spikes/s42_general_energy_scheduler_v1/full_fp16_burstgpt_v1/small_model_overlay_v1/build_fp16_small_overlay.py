#!/usr/bin/env python3
"""Bind ten Llama 1B requests to the immutable F16 BurstGPT replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FP16_ROOT = HERE.parent
S42_ROOT = FP16_ROOT.parent
REPO_ROOT = HERE.parents[4]
SOURCE_TRACE = (
    REPO_ROOT
    / "research_dev/spikes/s41_gemma_qwen_continuous_baseline"
    / "tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1"
    / "REQUESTS_SEMANTIC_SOURCE.jsonl"
)
DONOR_ROOT = S42_ROOT / "small_model_overlay_v1"
DONOR_TRACE = DONOR_ROOT / "REQUESTS_BURSTGPT_LLAMA1B_84.jsonl"
DONOR_MANIFEST = DONOR_ROOT / "TRACE_MANIFEST.json"
OVERLAY_TRACE = HERE / "REQUESTS_LLAMA1B_10.jsonl"
MANIFEST = HERE / "TRACE_MANIFEST.json"

SOURCE_SCHEMA = "s41-gemma-qwen-request-semantic-source-v1"
SOURCE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4c"
    "31a4a9e8b4f9c085b29a14f80250a0ff"
)
DONOR_SCHEMA = "s42-three-model-burstgpt-small-overlay-v1"
DONOR_SHA256 = (
    "2eb73767d0a4e766db7e4c5f7047ee14"
    "36821e94a045d2630d84f9edc38ef814"
)
TRACE_SCHEMA = "s42-full-fp16-llama1b-overlay-v1"
MANIFEST_SCHEMA = "s42-full-fp16-llama1b-overlay-manifest-v1"
OVERLAY_STREAM = "llama-3.2-1b-overlay"
LLAMA1 = "llama-3.2-1b-instruct-q4_0"
QWEN_F16 = "qwen3-14b-q4km-dequant-f16"
GEMMA_F16 = "gemma-4-12b-q40-dequant-f16"

MODEL_INVENTORY = {
    GEMMA_F16: {
        "artifact_bytes": 23_832_065_056,
        "artifact_file": "gemma-4-12B-Q40-dequant-f16.gguf",
        "artifact_sha256": (
            "ed76f2183d2d1d65091986033023e6c7"
            "8d27f6276c1b0c5826cc92acf73538cf"
        ),
        "kind": "text_decoder_f16_proxy",
    },
    LLAMA1: {
        "artifact_bytes": 770_928_288,
        "artifact_file": "Llama-3.2-1B-Instruct-Q4_0.gguf",
        "artifact_sha256": (
            "4b90b1d7ae7324676194755a6dfce11c"
            "b6e457982c4c01a1db2857be1ed064ad"
        ),
        "kind": "text_decoder",
    },
    QWEN_F16: {
        "artifact_bytes": 29_543_423_360,
        "artifact_file": "Qwen3-14B-Q4KM-dequant-f16.gguf",
        "artifact_sha256": (
            "d89e9e823744222e595e0b3c8fd5436c"
            "e5d3a6a446fa42492ebce6064dfa9718"
        ),
        "kind": "text_decoder_f16_proxy",
    },
}


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


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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


def load_object(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BuildError(f"cannot parse object: {path}") from error
    if type(value) is not dict or canonical(value) != content:
        raise BuildError(f"object is not canonical: {path}")
    return value, content


def derive() -> tuple[bytes, bytes]:
    source_rows, source_content = load_rows(SOURCE_TRACE)
    if (
        digest(source_content) != SOURCE_SHA256
        or len(source_rows) != 74
        or any(row.get("schema") != SOURCE_SCHEMA for row in source_rows)
        or sum(row["input_tokens"] for row in source_rows) != 33_843
        or sum(row["output_tokens"] for row in source_rows) != 11_605
    ):
        raise BuildError("F16 source trace identity mismatch")

    donor_rows, donor_content = load_rows(DONOR_TRACE)
    donor_manifest, donor_manifest_content = load_object(DONOR_MANIFEST)
    if (
        digest(donor_content) != DONOR_SHA256
        or donor_manifest.get("trace", {}).get("sha256") != DONOR_SHA256
        or len(donor_rows) != 84
        or any(row.get("schema") != DONOR_SCHEMA for row in donor_rows)
    ):
        raise BuildError("small-model donor identity mismatch")

    rows = [
        dict(row)
        for row in donor_rows
        if row.get("trace_stream_id") == OVERLAY_STREAM
    ]
    rows.sort(key=lambda row: row["stream_request_index"])
    if (
        len(rows) != 10
        or [row["stream_request_index"] for row in rows] != list(range(10))
    ):
        raise BuildError("small-model overlay geometry mismatch")
    for row in rows:
        row["combined_request_index"] = row.pop("mixed_request_index")
        row["overlay_request_index"] = row["stream_request_index"]
        row["schema"] = TRACE_SCHEMA
        if (
            row.get("execution_model_id") != LLAMA1
            or row.get("model_artifact_sha256")
            != MODEL_INVENTORY[LLAMA1]["artifact_sha256"]
            or row.get("model_artifact_bytes")
            != MODEL_INVENTORY[LLAMA1]["artifact_bytes"]
            or len(row.get("prompt_tokens", [])) != row.get("input_tokens")
        ):
            raise BuildError("small-model artifact or token mismatch")

    trace_content = b"".join(canonical(row) for row in rows)
    overlay_input_tokens = sum(row["input_tokens"] for row in rows)
    overlay_output_tokens = sum(row["output_tokens"] for row in rows)
    manifest = {
        "base_trace": {
            "input_tokens": 33_843,
            "output_tokens": 11_605,
            "path": str(SOURCE_TRACE.relative_to(REPO_ROOT)),
            "record_count": 74,
            "schema": SOURCE_SCHEMA,
            "sha256": SOURCE_SHA256,
        },
        "combined_work": {
            "input_tokens": 33_843 + overlay_input_tokens,
            "output_tokens": 11_605 + overlay_output_tokens,
            "record_count": 84,
        },
        "derivation": {
            "donor_manifest_sha256": digest(donor_manifest_content),
            "donor_trace_path": str(DONOR_TRACE.relative_to(REPO_ROOT)),
            "donor_trace_sha256": DONOR_SHA256,
        },
        "execution_models": {
            GEMMA_F16: 17,
            LLAMA1: 10,
            QWEN_F16: 57,
        },
        "model_inventory": MODEL_INVENTORY,
        "overlay_trace": {
            "arrival_first_us": min(row["arrival_us"] for row in rows),
            "arrival_last_us": max(row["arrival_us"] for row in rows),
            "input_tokens": overlay_input_tokens,
            "output_tokens": overlay_output_tokens,
            "path": str(OVERLAY_TRACE.relative_to(REPO_ROOT)),
            "record_count": len(rows),
            "schema": TRACE_SCHEMA,
            "sha256": digest(trace_content),
            "stream_id": OVERLAY_STREAM,
        },
        "schema": MANIFEST_SCHEMA,
    }
    if manifest["combined_work"] != {
        "input_tokens": 38_948,
        "output_tokens": 13_132,
        "record_count": 84,
    }:
        raise BuildError("combined work geometry mismatch")
    return trace_content, canonical(manifest)


def write_or_compare(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise BuildError(f"existing output differs: {path}")
        return
    path.write_bytes(content)


def main() -> int:
    trace_content, manifest_content = derive()
    write_or_compare(OVERLAY_TRACE, trace_content)
    write_or_compare(MANIFEST, manifest_content)
    manifest = json.loads(manifest_content)
    print(json.dumps({
        "combined_work": manifest["combined_work"],
        "manifest": str(MANIFEST),
        "overlay_sha256": manifest["overlay_trace"]["sha256"],
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
