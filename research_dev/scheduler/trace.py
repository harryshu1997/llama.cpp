"""Strict JSONL adapters for scheduler request traces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ._internal.policy import QUALITY_RANK, Request, SchedulerError


__all__ = [
    "AdaptedTrace",
    "BURSTGPT_TRACE_SCHEMAS",
    "MIXED_TRACE_SCHEMA",
    "MIXED_TRACE_SCHEMAS",
    "SEMANTIC_SOURCE_TRACE_SCHEMA",
    "SMALL_MODEL_OVERLAY_TRACE_SCHEMA",
    "SOURCE_TRACE_SCHEMA",
    "TraceError",
    "load_burstgpt_trace",
    "load_mixed_model_trace",
    "load_trace",
    "sha256_file",
]


SOURCE_TRACE_SCHEMA = "s41-gemma-qwen-request-semantic-long-v1"
SEMANTIC_SOURCE_TRACE_SCHEMA = "s41-gemma-qwen-request-semantic-source-v1"
BURSTGPT_TRACE_SCHEMAS = frozenset({
    SOURCE_TRACE_SCHEMA,
    SEMANTIC_SOURCE_TRACE_SCHEMA,
})
MIXED_TRACE_SCHEMA = "s42-six-model-burstgpt-mixed-v1"
SMALL_MODEL_OVERLAY_TRACE_SCHEMA = (
    "s42-three-model-burstgpt-small-overlay-v1"
)
MIXED_TRACE_SCHEMAS = frozenset({
    MIXED_TRACE_SCHEMA,
    SMALL_MODEL_OVERLAY_TRACE_SCHEMA,
})


class TraceError(ValueError):
    pass


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TraceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TraceError(f"cannot hash trace: {exc}") from exc
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class AdaptedTrace:
    source_path: str
    source_sha256: str
    requests: tuple[Request, ...]
    output_tokens: int


def _quality(value: str) -> str:
    if value not in QUALITY_RANK:
        raise TraceError("unknown trace quality requirement")
    return value


def load_burstgpt_trace(
    path: Path,
    workload_map: Mapping[str, str],
    quality_requirement: str = "approximate",
) -> AdaptedTrace:
    quality_requirement = _quality(quality_requirement)
    requests: list[Request] = []
    seen: set[str] = set()
    previous: tuple[int, str] | None = None
    try:
        with path.open("r", encoding="ascii") as source:
            for line_number, raw_line in enumerate(source, 1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(
                        raw_line, object_pairs_hook=_no_duplicates
                    )
                except json.JSONDecodeError as exc:
                    raise TraceError(
                        f"line {line_number}: invalid JSON: {exc}"
                    ) from exc
                if (
                    type(row) is not dict
                    or row.get("schema") not in BURSTGPT_TRACE_SCHEMAS
                ):
                    raise TraceError(f"line {line_number}: schema mismatch")
                event_id = row.get("event_id")
                model_id = row.get("model_id")
                arrival_us = row.get("arrival_us")
                slo_us = row.get("slo_us")
                input_tokens = row.get("input_tokens")
                output_tokens = row.get("output_tokens")
                if (
                    type(event_id) is not str
                    or not event_id
                    or event_id in seen
                    or type(model_id) is not str
                    or model_id not in workload_map
                    or type(arrival_us) is not int
                    or arrival_us < 0
                    or type(slo_us) is not int
                    or slo_us <= 0
                    or type(input_tokens) is not int
                    or input_tokens <= 0
                    or type(output_tokens) is not int
                    or output_tokens <= 0
                ):
                    raise TraceError(f"line {line_number}: invalid request")
                key = (arrival_us, event_id)
                if previous is not None and key < previous:
                    raise TraceError("requests are not in arrival order")
                previous = key
                seen.add(event_id)
                request = Request(
                    request_id=event_id,
                    workload_id=workload_map[model_id],
                    arrival_us=arrival_us,
                    deadline_us=arrival_us + slo_us,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    quality_requirement=quality_requirement,
                )
                try:
                    request.validate()
                except SchedulerError as exc:
                    raise TraceError(
                        f"line {line_number}: invalid request: {exc}"
                    ) from exc
                requests.append(request)
    except (OSError, UnicodeError) as exc:
        raise TraceError(f"cannot read trace: {exc}") from exc
    if not requests:
        raise TraceError("trace is empty")
    return AdaptedTrace(
        source_path=str(path),
        source_sha256=sha256_file(path),
        requests=tuple(requests),
        output_tokens=sum(request.output_tokens for request in requests),
    )


def load_mixed_model_trace(
    path: Path,
    workload_map: Mapping[str, str],
    quality_requirement: str = "approximate",
) -> AdaptedTrace:
    quality_requirement = _quality(quality_requirement)
    requests: list[Request] = []
    seen: set[str] = set()
    previous: tuple[int, str] | None = None
    try:
        with path.open("r", encoding="ascii") as source:
            for line_number, raw_line in enumerate(source, 1):
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line, object_pairs_hook=_no_duplicates)
                if (
                    type(row) is not dict
                    or row.get("schema") not in MIXED_TRACE_SCHEMAS
                ):
                    raise TraceError(f"line {line_number}: schema mismatch")
                event_id = row.get("event_id")
                model_id = row.get("execution_model_id")
                arrival_us = row.get("arrival_us")
                slo_us = row.get("slo_us")
                input_tokens = row.get("input_tokens")
                output_tokens = row.get("output_tokens")
                if (
                    type(event_id) is not str
                    or not event_id
                    or event_id in seen
                    or type(model_id) is not str
                    or model_id not in workload_map
                    or type(arrival_us) is not int
                    or arrival_us < 0
                    or type(slo_us) is not int
                    or slo_us <= 0
                    or type(input_tokens) is not int
                    or input_tokens <= 0
                    or type(output_tokens) is not int
                    or output_tokens <= 0
                ):
                    raise TraceError(f"line {line_number}: invalid request")
                key = (arrival_us, event_id)
                if previous is not None and key < previous:
                    raise TraceError("requests are not in arrival order")
                previous = key
                seen.add(event_id)
                features = {
                    "image_bytes": int(row.get("image_bytes", 0)),
                    "image_count": int(row.get("image_count", 0)),
                    "image_pixels": int(row.get("image_pixels", 0)),
                    "is_multimodal": int(
                        row.get("prompt_transport") == "multimodal_message"
                    ),
                    "vision_projector_bytes": int(
                        row.get("vision_projector_bytes", 0)
                    ),
                }
                request = Request(
                    request_id=event_id,
                    workload_id=workload_map[model_id],
                    arrival_us=arrival_us,
                    deadline_us=arrival_us + slo_us,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    quality_requirement=quality_requirement,
                    features=features,
                )
                request.validate()
                requests.append(request)
    except TraceError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        SchedulerError,
    ) as exc:
        raise TraceError(f"cannot adapt mixed trace: {exc}") from exc
    if not requests:
        raise TraceError("trace is empty")
    return AdaptedTrace(
        source_path=str(path),
        source_sha256=sha256_file(path),
        requests=tuple(requests),
        output_tokens=sum(request.output_tokens for request in requests),
    )


def load_trace(
    path: Path,
    workload_map: Mapping[str, str],
    quality_requirement: str = "approximate",
) -> AdaptedTrace:
    try:
        with path.open("r", encoding="ascii") as source:
            first = next(line for line in source if line.strip())
        row = json.loads(first, object_pairs_hook=_no_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, StopIteration) as exc:
        raise TraceError(f"cannot detect trace schema: {exc}") from exc
    schema = row.get("schema") if type(row) is dict else None
    if schema in BURSTGPT_TRACE_SCHEMAS:
        return load_burstgpt_trace(
            path, workload_map, quality_requirement
        )
    if schema in MIXED_TRACE_SCHEMAS:
        return load_mixed_model_trace(
            path, workload_map, quality_requirement
        )
    raise TraceError("unsupported trace schema")
