#!/usr/bin/env python3
"""Verify the focused BurstGPT plus Llama 1B trace derivation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_small_model_overlay import (
    MANIFEST,
    MANIFEST_SCHEMA,
    OVERLAY_MODEL,
    TRACE,
    TRACE_SCHEMA,
    BuildError,
    derive,
)


class TraceValidationError(ValueError):
    pass


def validate(
    trace_path: Path = TRACE,
    manifest_path: Path = MANIFEST,
) -> dict[str, Any]:
    try:
        expected_trace, expected_manifest = derive()
        trace_content = trace_path.read_bytes()
        manifest_content = manifest_path.read_bytes()
    except (BuildError, OSError) as error:
        raise TraceValidationError(str(error)) from error
    if trace_content != expected_trace:
        raise TraceValidationError("trace differs from deterministic derivation")
    if manifest_content != expected_manifest:
        raise TraceValidationError(
            "manifest differs from deterministic derivation"
        )
    manifest = json.loads(manifest_content)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("trace", {}).get("schema") != TRACE_SCHEMA
        or manifest.get("record_count") != 84
        or manifest.get("models", {}).get(OVERLAY_MODEL) != 10
    ):
        raise TraceValidationError("derived trace metadata mismatch")
    return {
        "models": manifest["models"],
        "records": manifest["record_count"],
        "trace_sha256": manifest["trace"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    try:
        result = validate(args.trace, args.manifest)
    except TraceValidationError as error:
        raise SystemExit(f"FAIL: {error}") from error
    print(json.dumps({"verdict": "PASS", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
