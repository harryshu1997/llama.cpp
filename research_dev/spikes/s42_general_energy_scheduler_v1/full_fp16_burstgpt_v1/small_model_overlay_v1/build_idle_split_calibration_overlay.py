#!/usr/bin/env python3
"""Build an idle-only Llama FFN calibration overlay for OP15 sharing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import build_phase_calibration_overlay as phase


HERE = Path(__file__).resolve().parent
TRACE = HERE / "REQUESTS_LLAMA1B_IDLE_SPLIT_CALIBRATION_40.jsonl"
MANIFEST = HERE / "TRACE_IDLE_SPLIT_CALIBRATION_MANIFEST.json"
TRACE_SCHEMA = "s42-full-fp16-llama1b-idle-split-calibration-v1"
MANIFEST_SCHEMA = (
    "s42-full-fp16-llama1b-idle-split-calibration-manifest-v1"
)
STREAM_ID = "llama-3.2-1b-idle-split-calibration"


class BuildError(ValueError):
    pass


def canonical(value: Any) -> bytes:
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


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def derive() -> tuple[bytes, bytes]:
    source_trace, source_manifest_raw = phase.derive()
    if (
        source_trace != phase.TRACE.read_bytes()
        or source_manifest_raw != phase.MANIFEST.read_bytes()
    ):
        raise BuildError("phase overlay is not reproducible")
    source_rows = []
    for line in source_trace.splitlines():
        row = json.loads(line)
        if row["target_large_phase"] == "idle":
            source_rows.append(row)
    if len(source_rows) != 10:
        raise BuildError("idle donor geometry")

    rows = []
    for repeat in range(4):
        for donor_index, donor in enumerate(source_rows):
            index = len(rows)
            phase_offset_us = 1_000_000 + index * 1_000_000
            row = dict(donor)
            row.update({
                "arrival_order": index,
                "arrival_us": 3_000_000_000 + phase_offset_us,
                "combined_request_index": 74 + index,
                "event_id": (
                    f"s42:llama1b-idle-split-calibration:{repeat}:"
                    f"{donor_index:02d}"
                ),
                "overlay_request_index": index,
                "phase_offset_us": phase_offset_us,
                "schema": TRACE_SCHEMA,
                "stream_request_index": index,
                "trace_stream_id": STREAM_ID,
            })
            rows.append(row)

    trace = b"".join(canonical(row) for row in rows)
    source_manifest = json.loads(source_manifest_raw)
    input_tokens = sum(row["input_tokens"] for row in rows)
    output_tokens = sum(row["output_tokens"] for row in rows)
    manifest = {
        "arrival_contract": {
            "anchor": "observed_idle_large_model_phase_event",
            "phase_offset_field": "phase_offset_us",
            "phases": ["idle"],
            "purpose": "OP15 idle HTP split calibration",
        },
        "base_trace": source_manifest["base_trace"],
        "combined_work": {
            "input_tokens": 33_843 + input_tokens,
            "output_tokens": 11_605 + output_tokens,
            "record_count": 74 + len(rows),
        },
        "derivation": {
            "donor_repeats": 4,
            "source_phase_manifest_sha256": digest(source_manifest_raw),
            "source_phase_trace_sha256": digest(source_trace),
        },
        "execution_models": {
            **source_manifest["execution_models"],
            phase.base.LLAMA1: len(rows),
        },
        "model_inventory": source_manifest["model_inventory"],
        "overlay_trace": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "path": str(TRACE.relative_to(phase.base.REPO_ROOT)),
            "record_count": len(rows),
            "schema": TRACE_SCHEMA,
            "sha256": digest(trace),
            "stream_id": STREAM_ID,
        },
        "schema": MANIFEST_SCHEMA,
    }
    if len(rows) != 40:
        raise BuildError("idle split overlay count")
    return trace, canonical(manifest)


def write_or_compare(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise BuildError(f"existing output differs: {path}")
        return
    path.write_bytes(content)


def main() -> int:
    trace, manifest = derive()
    write_or_compare(TRACE, trace)
    write_or_compare(MANIFEST, manifest)
    print(json.dumps({
        "manifest": str(MANIFEST),
        "record_count": 40,
        "status": "PASS",
        "trace_sha256": digest(trace),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
