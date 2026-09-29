#!/usr/bin/env python3
"""Compatibility adapters for scheduler-owned trace parsing."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    AdaptedTrace,
    MIXED_TRACE_SCHEMA,
    SOURCE_TRACE_SCHEMA,
    TraceError,
    load_burstgpt_trace,
    load_mixed_model_trace as _load_mixed_model_trace,
    load_trace as _load_trace,
    sha256_file,
)


SOURCE_SCHEMA = SOURCE_TRACE_SCHEMA
MIXED_SCHEMA = MIXED_TRACE_SCHEMA
load_burstgpt = load_burstgpt_trace


def _validate_mixed(path: Path) -> None:
    try:
        from mixed_model_trace_v1.verify_mixed_trace import (
            TraceValidationError,
            validate,
        )
    except ImportError as exc:
        raise TraceError(f"mixed trace validation failed: {exc}") from exc
    try:
        validate(path, path.parent / "TRACE_MANIFEST.json")
    except TraceValidationError as exc:
        raise TraceError(f"mixed trace validation failed: {exc}") from exc


def load_mixed_model_trace(
    path: Path,
    workload_map: Mapping[str, str],
    quality_requirement: str = "approximate",
) -> AdaptedTrace:
    _validate_mixed(path)
    return _load_mixed_model_trace(
        path, workload_map, quality_requirement
    )


def load_trace(
    path: Path,
    workload_map: Mapping[str, str],
    quality_requirement: str = "approximate",
) -> AdaptedTrace:
    try:
        with path.open("r", encoding="ascii") as source:
            first = next(line for line in source if line.strip())
        row = json.loads(first)
    except (OSError, UnicodeError, json.JSONDecodeError, StopIteration) as exc:
        raise TraceError(f"cannot detect trace schema: {exc}") from exc
    if type(row) is dict and row.get("schema") == MIXED_SCHEMA:
        return load_mixed_model_trace(
            path, workload_map, quality_requirement
        )
    return _load_trace(path, workload_map, quality_requirement)


__all__ = [
    "AdaptedTrace",
    "MIXED_SCHEMA",
    "SOURCE_SCHEMA",
    "TraceError",
    "load_burstgpt",
    "load_mixed_model_trace",
    "load_trace",
    "sha256_file",
]
