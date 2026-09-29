#!/usr/bin/env python3
"""Build a phase-agnostic Llama 1B validation overlay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import build_fp16_small_overlay as base


HERE = Path(__file__).resolve().parent
TRACE = HERE / "REQUESTS_LLAMA1B_NATURAL_VALIDATION_40.jsonl"
MANIFEST = HERE / "TRACE_NATURAL_VALIDATION_MANIFEST.json"
TRACE_SCHEMA = "s42-full-fp16-llama1b-natural-validation-v1"
MANIFEST_SCHEMA = (
    "s42-full-fp16-llama1b-natural-validation-manifest-v1"
)
STREAM_ID = "llama-3.2-1b-natural-validation"
FIRST_US = 30_000_000
LAST_US = 3_100_000_000
REPEATS = 4


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


def van_der_corput(n: int) -> tuple[int, int]:
    if type(n) is not int or n <= 0:
        raise BuildError("van der Corput index")
    numerator = 0
    denominator = 1
    while n:
        n, bit = divmod(n, 2)
        numerator = numerator * 2 + bit
        denominator *= 2
    return numerator, denominator


def arrival_schedule(count: int) -> list[int]:
    if type(count) is not int or count <= 0:
        raise BuildError("natural validation count")
    span = LAST_US - FIRST_US
    arrivals = []
    for index in range(1, count + 1):
        numerator, denominator = van_der_corput(index)
        arrivals.append(
            FIRST_US + (span * numerator + denominator // 2) // denominator
        )
    arrivals.sort()
    if len(set(arrivals)) != count:
        raise BuildError("natural validation arrival collision")
    return arrivals


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

    count = len(donors) * REPEATS
    arrivals = arrival_schedule(count)
    rows = []
    for index, arrival_us in enumerate(arrivals):
        donor_index = index % len(donors)
        donor = donors[donor_index]
        row = dict(donor)
        row.update({
            "arrival_order": index,
            "arrival_us": arrival_us,
            "combined_request_index": 74 + index,
            "event_id": f"s42:llama1b-natural-validation:{index:02d}",
            "overlay_request_index": index,
            "schema": TRACE_SCHEMA,
            "source_overlay_event_id": donor["event_id"],
            "source_overlay_request_index": donor_index,
            "stream_request_index": index,
            "trace_stream_id": STREAM_ID,
        })
        rows.append(row)

    trace = b"".join(canonical(row) for row in rows)
    base_manifest = json.loads(base_manifest_raw)
    input_tokens = sum(row["input_tokens"] for row in rows)
    output_tokens = sum(row["output_tokens"] for row in rows)
    manifest = {
        "arrival_contract": {
            "generator": "van_der_corput_base2",
            "interval_us": [FIRST_US, LAST_US],
            "phase_labels_used": False,
            "purpose": "natural holdout across the full trace horizon",
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
            "donor_repeats": REPEATS,
        },
        "execution_models": {
            base.GEMMA_F16: 17,
            base.LLAMA1: len(rows),
            base.QWEN_F16: 57,
        },
        "model_inventory": base.MODEL_INVENTORY,
        "overlay_trace": {
            "arrival_first_us": min(arrivals),
            "arrival_last_us": max(arrivals),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "path": str(TRACE.relative_to(base.REPO_ROOT)),
            "record_count": len(rows),
            "schema": TRACE_SCHEMA,
            "sha256": digest(trace),
            "stream_id": STREAM_ID,
        },
        "schema": MANIFEST_SCHEMA,
    }
    if len(rows) != 40:
        raise BuildError("natural validation overlay count")
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
