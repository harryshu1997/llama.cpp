#!/usr/bin/env python3
"""Build a phase-anchored Llama 1B calibration and holdout overlay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import build_fp16_small_overlay as base


HERE = Path(__file__).resolve().parent
TRACE = HERE / "REQUESTS_LLAMA1B_PHASE_CALIBRATION_40.jsonl"
MANIFEST = HERE / "TRACE_PHASE_CALIBRATION_MANIFEST.json"
TRACE_SCHEMA = "s42-full-fp16-llama1b-phase-overlay-v1"
MANIFEST_SCHEMA = "s42-full-fp16-llama1b-phase-overlay-manifest-v1"
PHASES = ("qwen", "switching", "gemma", "idle")
PHASE_NOMINAL_START_US = {
    "qwen": 0,
    "switching": 1_900_000_000,
    "gemma": 2_000_000_000,
    "idle": 3_000_000_000,
}
TRAIN_DONORS = frozenset({0, 1, 2, 4, 5, 7, 9})


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
    base_trace, base_manifest_raw = base.derive()
    if (
        base_trace != base.OVERLAY_TRACE.read_bytes()
        or base_manifest_raw != base.MANIFEST.read_bytes()
    ):
        raise BuildError("base overlay is not reproducible")
    donors = [json.loads(line) for line in base_trace.splitlines()]
    if (
        len(donors) != 10
        or [row["overlay_request_index"] for row in donors]
            != list(range(10))
    ):
        raise BuildError("base donor geometry")

    rows = []
    for phase_index, phase in enumerate(PHASES):
        for donor_index, donor in enumerate(donors):
            index = len(rows)
            row = dict(donor)
            phase_offset_us = 1_000_000 + donor_index * 1_000_000
            row.update({
                "arrival_order": index,
                "arrival_us": (
                    PHASE_NOMINAL_START_US[phase] + phase_offset_us
                ),
                "calibration_split": (
                    "train" if donor_index in TRAIN_DONORS else "holdout"
                ),
                "combined_request_index": 74 + index,
                "event_id": (
                    f"s42:llama1b-phase-calibration:{phase}:"
                    f"{donor_index:02d}"
                ),
                "overlay_request_index": index,
                "phase_offset_us": phase_offset_us,
                "schema": TRACE_SCHEMA,
                "source_overlay_event_id": donor["event_id"],
                "source_overlay_request_index": donor_index,
                "stream_request_index": index,
                "target_large_phase": phase,
                "target_large_phase_id": phase_index + 1,
                "trace_stream_id": "llama-3.2-1b-phase-calibration",
            })
            rows.append(row)

    trace = b"".join(canonical(row) for row in rows)
    base_manifest = json.loads(base_manifest_raw)
    input_tokens = sum(row["input_tokens"] for row in rows)
    output_tokens = sum(row["output_tokens"] for row in rows)
    manifest = {
        "arrival_contract": {
            "anchor": "observed_large_model_phase_event",
            "phase_offset_field": "phase_offset_us",
            "phases": list(PHASES),
            "purpose": "phase-stratified calibration, not natural arrivals",
        },
        "base_trace": base_manifest["base_trace"],
        "combined_work": {
            "input_tokens": 33_843 + input_tokens,
            "output_tokens": 11_605 + output_tokens,
            "record_count": 74 + len(rows),
        },
        "derivation": {
            "base_overlay_manifest_sha256": digest(base_manifest_raw),
            "base_overlay_sha256": digest(base_trace),
            "holdout_donor_indices": sorted(set(range(10)) - TRAIN_DONORS),
            "train_donor_indices": sorted(TRAIN_DONORS),
        },
        "execution_models": {
            base.GEMMA_F16: 17,
            base.LLAMA1: len(rows),
            base.QWEN_F16: 57,
        },
        "model_inventory": base.MODEL_INVENTORY,
        "overlay_trace": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "path": str(TRACE.relative_to(base.REPO_ROOT)),
            "record_count": len(rows),
            "schema": TRACE_SCHEMA,
            "sha256": digest(trace),
            "stream_id": "llama-3.2-1b-phase-calibration",
        },
        "schema": MANIFEST_SCHEMA,
    }
    if len(rows) != 40:
        raise BuildError("phase overlay count")
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
