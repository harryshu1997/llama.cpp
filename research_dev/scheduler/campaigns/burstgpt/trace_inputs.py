"""Validate BurstGPT inputs and construct deterministic replay schedules."""

import hashlib
from pathlib import Path
from typing import Any

from .common import UnifiedTraceError, canonical, digest, load_rows, require


REPLAY_SCHEDULE_SCHEMA = "research-scheduler-burstgpt-replay-v1"
REPLAY_START_US = 1_000_000
QWEN_ROLE = "hot"
GEMMA_ROLE = "cold"
TRACE_HOT_MODEL_ID = "gemma-4-12b-it-q8_0"
TRACE_COLD_MODEL_ID = "qwen3-14b-q4_k_m"


def trace_role(row: dict[str, Any]) -> str:
    if row.get("model_id") == TRACE_HOT_MODEL_ID:
        return QWEN_ROLE
    if row.get("model_id") == TRACE_COLD_MODEL_ID:
        return GEMMA_ROLE
    raise UnifiedTraceError("unexpected source model")


def validate_trace(
    large_path: Path,
    overlay_path: Path,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    large = load_rows(large_path)
    base_identity = manifest.get("base_trace")
    overlay_identity = manifest.get("overlay_trace")
    # A trace may carry no small-model overlay; the manifest says so with record_count 0 and the
    # file is empty. load_rows rejects empty files, so honor that case here.
    overlay_empty = (
        type(overlay_identity) is dict
        and overlay_identity.get("record_count") == 0
        and overlay_path.stat().st_size == 0
    )
    overlay = [] if overlay_empty else load_rows(overlay_path)
    # The trace identity comes from the manifest: record counts per file, per-role counts of the
    # large file, and the combined count. The original unified trace (74 + 10 rows, 57 hot / 17 cold)
    # predates the per-role fields, so its manifest is accepted by its record count alone.
    combined_work = manifest.get("combined_work")
    legacy = (
        type(base_identity) is dict
        and base_identity.get("record_count") == 74
        and "roles" not in base_identity
    )
    expected_roles = (
        {QWEN_ROLE: 57, GEMMA_ROLE: 17} if legacy
        else (base_identity.get("roles") if type(base_identity) is dict else None)
    )
    inventory = manifest.get("model_inventory")
    require(
        type(inventory) is dict
        and len(inventory) > 0
        and all(
            type(entry) is dict
            and type(entry.get("artifact_bytes")) is int
            and entry["artifact_bytes"] > 0
            and type(entry.get("artifact_sha256")) is str
            and len(entry["artifact_sha256"]) == 64
            and type(entry.get("artifact_file")) is str
            for entry in inventory.values()
        ),
        "trace manifest model_inventory",
    )
    require(
        manifest.get("schema")
            == "s42-full-fp16-llama1b-overlay-manifest-v1"
        and type(base_identity) is dict
        and type(overlay_identity) is dict
        and type(combined_work) is dict
        and base_identity.get("sha256") == digest(large_path)
        and overlay_identity.get("sha256") == digest(overlay_path)
        and base_identity.get("record_count") == len(large)
        and overlay_identity.get("record_count") == len(overlay)
        and combined_work.get("record_count") == len(large) + len(overlay)
        and len(large) > 0
        and type(expected_roles) is dict
        and set(expected_roles) == {QWEN_ROLE, GEMMA_ROLE}
        and sum(trace_role(row) == QWEN_ROLE for row in large) == expected_roles[QWEN_ROLE]
        and sum(trace_role(row) == GEMMA_ROLE for row in large) == expected_roles[GEMMA_ROLE]
        and all(
            len(row.get("prompt_tokens", [])) == row.get("input_tokens")
            and type(row.get("output_tokens")) is int
            and row["output_tokens"] > 0
            and row.get("slo_us") == 30_000_000
            for row in (*large, *overlay)
        ),
        "unified trace identity",
    )
    return large, overlay


def merge_rows(
    large: list[dict[str, Any]],
    overlay: list[dict[str, Any]],
    model_ids: dict[str, str],
) -> list[dict[str, Any]]:
    merged = []
    for row in large:
        role = trace_role(row)
        merged.append({
            "model_id": model_ids[role],
            "row": row,
            "source": "large",
            "source_index": row["request_index"],
        })
    for row in overlay:
        merged.append({
            "model_id": row["execution_model_id"],
            "row": row,
            "source": "overlay",
            "source_index": row["overlay_request_index"],
        })
    merged.sort(key=lambda item: (
        item["row"]["arrival_us"],
        0 if item["source"] == "large" else 1,
        item["source_index"],
    ))
    require(
        len(merged) == len(large) + len(overlay)
        and all(
            item["row"].get("combined_request_index") == index
            for index, item in enumerate(merged)
            if item["source"] == "overlay"
        ),
        "unified trace merge identity",
    )
    for combined_index, item in enumerate(merged):
        item["combined_index"] = combined_index
    return merged


def select_rows(
    merged: list[dict[str, Any]], request_indices: str | None
) -> list[dict[str, Any]]:
    if request_indices is None:
        return merged
    fields = request_indices.split(",")
    require(
        fields
        and all(field.isascii() and field.isdigit() for field in fields),
        "request indices",
    )
    indices = [int(field) for field in fields]
    require(
        len(indices) == len(set(indices))
        and indices == sorted(indices)
        and all(0 <= index < len(merged) for index in indices),
        "request indices",
    )
    return [merged[index] for index in indices]


def scale_replay_arrivals(
    selected: list[dict[str, Any]], arrival_scale: int | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(bool(selected), "replay schedule is empty")
    require(
        arrival_scale is None
        or (type(arrival_scale) is int and arrival_scale > 0),
        "arrival scale",
    )
    source_first = selected[0]["row"]["arrival_us"]
    require(
        type(source_first) is int and source_first >= 0,
        "source arrival",
    )
    scaled = []
    schedule = []
    previous_source = -1
    previous_replay = -1
    for item in selected:
        source_arrival_us = item["row"]["arrival_us"]
        require(
            type(source_arrival_us) is int
            and source_arrival_us >= previous_source,
            "source arrival order",
        )
        replay_arrival_us = (
            source_arrival_us
            if arrival_scale is None
            else REPLAY_START_US
                + arrival_scale * (source_arrival_us - source_first)
        )
        require(
            replay_arrival_us >= previous_replay,
            "replay arrival order",
        )
        row = dict(item["row"])
        row["arrival_us"] = replay_arrival_us
        row["source_arrival_us"] = source_arrival_us
        row["replay_arrival_us"] = replay_arrival_us
        scaled.append({**item, "row": row})
        schedule.append({
            "combined_request_index": item["combined_index"],
            "replay_arrival_us": replay_arrival_us,
            "request_id": row["event_id"],
            "source_arrival_us": source_arrival_us,
        })
        previous_source = source_arrival_us
        previous_replay = replay_arrival_us
    schedule_sha256 = "sha256:" + hashlib.sha256(
        canonical(schedule)
    ).hexdigest()
    source_last = schedule[-1]["source_arrival_us"]
    replay_first = schedule[0]["replay_arrival_us"]
    replay_last = schedule[-1]["replay_arrival_us"]
    return scaled, {
        "arrival_scale": arrival_scale,
        "replay_first_arrival_us": replay_first,
        "replay_last_arrival_us": replay_last,
        "replay_span_us": replay_last - replay_first,
        "schedule": schedule,
        "schedule_sha256": schedule_sha256,
        "schema": REPLAY_SCHEDULE_SCHEMA,
        "selected_indices": [
            item["combined_index"] for item in selected
        ],
        "source_first_arrival_us": source_first,
        "source_last_arrival_us": source_last,
        "source_span_us": source_last - source_first,
    }


def apply_named_replay_schedule(
    merged: list[dict[str, Any]], value: object
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(
        type(value) is dict
        and value.get("schema") == REPLAY_SCHEDULE_SCHEMA,
        "named replay schedule schema",
    )
    name = value.get("trace_name")
    arrivals = value.get("arrivals")
    require(
        type(name) is str and name and name.isascii()
        and type(arrivals) is list and bool(arrivals),
        "named replay schedule identity",
    )
    by_index = {item["combined_index"]: item for item in merged}
    selected = []
    schedule = []
    seen = set()
    previous_replay_us = -1
    for row in arrivals:
        require(type(row) is dict, "named replay arrival")
        combined_index = row.get("combined_request_index")
        replay_arrival_us = row.get("replay_arrival_us")
        require(
            type(combined_index) is int
            and combined_index in by_index
            and combined_index not in seen
            and type(replay_arrival_us) is int
            and replay_arrival_us >= previous_replay_us,
            "named replay arrival identity",
        )
        item = by_index[combined_index]
        source_arrival_us = item["row"]["arrival_us"]
        replay_row = dict(item["row"])
        replay_row["arrival_us"] = replay_arrival_us
        replay_row["source_arrival_us"] = source_arrival_us
        replay_row["replay_arrival_us"] = replay_arrival_us
        selected.append({**item, "row": replay_row})
        schedule.append({
            "combined_request_index": combined_index,
            "replay_arrival_us": replay_arrival_us,
            "request_id": replay_row["event_id"],
            "source_arrival_us": source_arrival_us,
        })
        seen.add(combined_index)
        previous_replay_us = replay_arrival_us
    schedule_sha256 = "sha256:" + hashlib.sha256(
        canonical(schedule)
    ).hexdigest()
    source_arrivals = tuple(
        row["source_arrival_us"] for row in schedule
    )
    return selected, {
        "arrival_scale": None,
        "replay_first_arrival_us": schedule[0]["replay_arrival_us"],
        "replay_last_arrival_us": schedule[-1]["replay_arrival_us"],
        "replay_span_us": (
            schedule[-1]["replay_arrival_us"]
            - schedule[0]["replay_arrival_us"]
        ),
        "schedule": schedule,
        "schedule_sha256": schedule_sha256,
        "schema": REPLAY_SCHEDULE_SCHEMA,
        "selected_indices": [
            row["combined_request_index"] for row in schedule
        ],
        "source_first_arrival_us": min(source_arrivals),
        "source_last_arrival_us": max(source_arrivals),
        "source_span_us": max(source_arrivals) - min(source_arrivals),
        "trace_name": name,
    }


def persist_replay_schedule(
    path: Path, replay_schedule: dict[str, Any]
) -> None:
    path.write_bytes(canonical(replay_schedule))
