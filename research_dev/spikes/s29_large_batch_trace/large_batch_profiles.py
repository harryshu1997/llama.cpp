#!/usr/bin/env python3
"""Derive and load S29 route profiles from one physical calibration artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
S26 = HERE.parent / "s26_priority_scheduler"
if str(S26) not in sys.path:
    sys.path.insert(0, str(S26))

from priority_policy import PriorityRoute, RouteBatchPoint


SCHEMA = "s29-large-batch-route-points-v1"
CALIBRATION_SCHEMA = "s29-route-calibration-v1"
EXPECTED_BATCHES = (1, 4, 24, 32)
ROUTE_BATCHES = {
    "R0": (1, 4, 24, 32),
    "R2": (1, 24, 32),
}
ACTIVE = {
    "R0": {"cuda-prefix", "cuda-mid", "cuda-tail"},
    "R2": {"op12-prefix", "op15-mid", "cuda-tail"},
}


class ProfileError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProfileError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"), object_pairs_hook=_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot load {path}: {exc}") from exc
    if type(value) is not dict:
        raise ProfileError(f"{path} must contain an object")
    return value


def _validate_calibration(value: dict[str, Any]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    if (
        value.get("schema") != CALIBRATION_SCHEMA
        or value.get("status") != "CALIBRATION_COMPLETE"
        or value.get("batches") != list(EXPECTED_BATCHES)
        or value.get("shape") != {"input_tokens": 1, "output_steps": 4, "context": 16}
        or type(value.get("reps")) is not int
        or value["reps"] < 2
    ):
        raise ProfileError("calibration header changed")
    capacities = value.get("capacities")
    if (
        type(capacities) is not dict
        or set(capacities) != {
            "cuda-prefix", "cuda-mid", "op12-prefix", "op15-mid", "cuda-tail",
        }
        or capacities["op12-prefix"] < 32
        or capacities["op15-mid"] < 32
        or capacities["cuda-prefix"] < 32
        or capacities["cuda-mid"] < 32
        or capacities["cuda-tail"] < 32
    ):
        raise ProfileError("calibration capacity is insufficient")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    points = value.get("points")
    if type(points) is not list:
        raise ProfileError("calibration points are missing")
    for row in points:
        if type(row) is not dict:
            raise ProfileError("calibration point must be an object")
        route = row.get("route_id")
        batch = row.get("batch_size")
        if route not in ACTIVE or batch not in EXPECTED_BATCHES:
            raise ProfileError("calibration point identity is invalid")
        if any(
            type(row.get(field)) is not int or row[field] <= 0
            for field in ("wall_us", "max_latency_us", "cuda_work_us")
        ):
            raise ProfileError("calibration timing is invalid")
        tokens = row.get("output_tokens")
        if type(tokens) is not list or len(tokens) != 4 or any(type(token) is not int for token in tokens):
            raise ProfileError("calibration output tokens are invalid")
        events = row.get("events")
        if type(events) is not dict or set(events) != set(capacities):
            raise ProfileError("calibration event set is invalid")
        for worker, worker_events in events.items():
            expected = 4 if worker in ACTIVE[route] else 0
            if type(worker_events) is not list or len(worker_events) != expected:
                raise ProfileError("calibration event count changed")
            if any(
                event.get("status") != "OK" or event.get("batch_size") != batch
                for event in worker_events
            ):
                raise ProfileError("calibration includes a failed or wrong-size batch")
        grouped.setdefault((route, batch), []).append(row)
    for route in ACTIVE:
        for batch in EXPECTED_BATCHES:
            rows = grouped.get((route, batch), [])
            if len(rows) != value["reps"]:
                raise ProfileError(f"{route} B{batch} replicate count changed")
    return grouped


def derive_bundle(calibration_path: Path, source_name: str | None = None) -> dict[str, Any]:
    calibration = load_json(calibration_path)
    grouped = _validate_calibration(calibration)
    digest = "sha256:" + sha256_file(calibration_path)
    capacities = dict(calibration["capacities"])
    routes = []
    for route_id, batches in ROUTE_BATCHES.items():
        points = []
        for batch in batches:
            rows = grouped[(route_id, batch)]
            duration = max(
                max(row["wall_us"], row["max_latency_us"]) for row in rows
            )
            duration = (duration * 11 + 9) // 10
            cuda_work = (
                min(row["cuda_work_us"] for row in rows)
                if route_id == "R0"
                else max(row["cuda_work_us"] for row in rows)
            )
            points.append({
                "batch_size": batch,
                "input_tokens": 1,
                "output_steps": 4,
                "duration_us": duration,
                "cuda_work_us": cuda_work,
                "evidence_sha256": [digest],
                "derivation": (
                    "1.10x max repeated wall/latency; conservative CUDA work "
                    + ("minimum baseline" if route_id == "R0" else "maximum treatment")
                ),
            })
        routes.append({
            "route_id": route_id,
            "resources": (
                ["cuda-prefix", "cuda-mid", "cuda-tail"]
                if route_id == "R0"
                else ["op12-prefix", "op15-mid", "cuda-tail"]
            ),
            "gather_us": 5000,
            "points": points,
        })
    return {
        "schema": SCHEMA,
        "scope": "MECHANICS_ONLY_F16_PHONE_Q8_SERVER_NUMERICALLY_UNCERTIFIED",
        "work": {"input_tokens": 1, "output_steps": 4, "synthetic_token": 2},
        "source": {
            "path": source_name or calibration_path.name,
            "sha256": digest,
        },
        "capacities": capacities,
        "urgent_reserve": {
            "cuda-prefix": 16,
            "cuda-mid": 16,
            "op12-prefix": 0,
            "op15-mid": 0,
            "cuda-tail": 0,
        },
        "routes": routes,
    }


def write_bundle(calibration_path: Path, output_path: Path) -> dict[str, Any]:
    value = derive_bundle(calibration_path, calibration_path.name)
    output_path.write_bytes(canonical_bytes(value))
    return value


def load_bundle(
    path: Path,
) -> tuple[tuple[PriorityRoute, ...], dict[str, int], dict[str, int], dict[str, Any]]:
    actual = load_json(path)
    source = actual.get("source")
    if type(source) is not dict or type(source.get("path")) is not str:
        raise ProfileError("profile source is missing")
    source_path = path.parent / source["path"]
    expected = derive_bundle(source_path, source["path"])
    if actual != expected or path.read_bytes() != canonical_bytes(expected):
        raise ProfileError("profile is not the canonical calibration derivative")
    routes = tuple(PriorityRoute(
        route_id=row["route_id"],
        resources=tuple(row["resources"]),
        points=tuple(RouteBatchPoint(
            batch_size=point["batch_size"],
            input_tokens=point["input_tokens"],
            output_steps=point["output_steps"],
            duration_us=point["duration_us"],
            cuda_work_us=point["cuda_work_us"],
            evidence_sha256=tuple(point["evidence_sha256"]),
            derivation=point["derivation"],
        ) for point in row["points"]),
        gather_us=row["gather_us"],
    ) for row in actual["routes"])
    return routes, dict(actual["capacities"]), dict(actual["urgent_reserve"]), actual
