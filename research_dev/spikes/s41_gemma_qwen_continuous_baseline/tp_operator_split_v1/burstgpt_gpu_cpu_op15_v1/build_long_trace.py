#!/usr/bin/env python3
"""Derive a bounded long-form trace from the frozen BurstGPT geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / "REQUESTS.jsonl"
OUTPUT = HERE / "REQUESTS_LONG.jsonl"
SOURCE_SHA256 = (
    "94c36fe3ac43281dc0c83a29a1519c7"
    "e72ac445ed9ded98d474b2231041c1735"
)
PREFILL_CAP = 512
DECODE_CAP = 32


def canonical(value: object) -> bytes:
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


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    source = SOURCE.read_bytes()
    if digest(source) != SOURCE_SHA256:
        raise RuntimeError("frozen source trace hash mismatch")

    output = bytearray()
    for index, raw in enumerate(source.splitlines(keepends=True)):
        if not raw.endswith(b"\n"):
            raise RuntimeError(f"source row {index} framing mismatch")
        row = json.loads(raw)
        if canonical(row) != raw:
            raise RuntimeError(f"source row {index} is not canonical")
        prompt = row["prompt_tokens"]
        input_tokens = min(row["source_input_tokens"], PREFILL_CAP)
        output_tokens = min(row["source_output_tokens"], DECODE_CAP)
        repeats = (input_tokens + len(prompt) - 1) // len(prompt)
        row["input_tokens"] = input_tokens
        row["output_tokens"] = output_tokens
        row["prompt_tokens"] = (prompt * repeats)[:input_tokens]
        row["schema"] = "s41-gemma-qwen-request-long-v1"
        output.extend(canonical(row))

    if OUTPUT.exists():
        if OUTPUT.read_bytes() != output:
            raise RuntimeError("existing long trace does not match derivation")
    else:
        OUTPUT.write_bytes(output)
    print(json.dumps({
        "decode_cap": DECODE_CAP,
        "output": str(OUTPUT),
        "prefill_cap": PREFILL_CAP,
        "records": len(output.splitlines()),
        "sha256": digest(bytes(output)),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
