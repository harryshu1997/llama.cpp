#!/usr/bin/env python3
"""Derive S31 route profiles from the selected-cut physical calibration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
S26 = HERE.parent / "s26_priority_scheduler"
if str(S26) not in sys.path:
    sys.path.insert(0, str(S26))

from cut_selector import (
    CutSelectionError,
    canonical_bytes,
    load_json,
    load_selection,
    sha256_file,
)
from priority_policy import PriorityRoute, RouteBatchPoint


SCHEMA = "s31-selected-route-points-v1"
CALIBRATION_SCHEMA = "s31-selected-route-calibration-v1"
EXPECTED_BATCHES = (1, 4, 24, 32)
ROUTE_BATCHES = {"R0": (1, 4, 24, 32), "R2": (1, 24, 32)}
ACTIVE = {
    "R0": {"cuda-prefix", "cuda-mid", "cuda-tail"},
    "R2": {"op12-prefix", "op15-mid", "cuda-tail"},
}
WORKERS = ("cuda-prefix", "cuda-mid", "op12-prefix", "op15-mid", "cuda-tail")


class ProfileError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        return load_json(path)
    except CutSelectionError as exc:
        raise ProfileError(str(exc)) from exc


def _positive(value: Any) -> bool:
    return type(value) is int and value > 0


def _validate_selection(calibration_path: Path, value: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    binding = value.get("selection")
    if (
        type(binding) is not dict
        or type(binding.get("path")) is not str
        or not binding["path"]
        or type(binding.get("sha256")) is not str
    ):
        raise ProfileError("calibration selection binding is missing")
    selection_path = calibration_path.parent / binding["path"]
    if binding["sha256"] != "sha256:" + sha256_file(selection_path):
        raise ProfileError("selection digest mismatch")
    try:
        selection = load_selection(selection_path)
    except CutSelectionError as exc:
        raise ProfileError(str(exc)) from exc
    cut = selection.get("selected", {}).get("cut_layer")
    if (
        selection.get("schema") != "s31-cut-selection-v1"
        or selection.get("status") != "CUT_SELECTED"
        or type(cut) is not int
        or not 1 <= cut < 8
        or value.get("selected_cut") != cut
    ):
        raise ProfileError("selected cut is invalid")
    return cut, selection


def _validate_calibration(
    path: Path, value: dict[str, Any],
) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], int, dict[str, Any]]:
    if (
        value.get("schema") != CALIBRATION_SCHEMA
        or value.get("status") != "CALIBRATION_COMPLETE"
        or value.get("batches") != list(EXPECTED_BATCHES)
        or value.get("shape") != {"input_tokens": 1, "output_steps": 4, "context": 16}
        or type(value.get("reps")) is not int
        or value["reps"] < 2
        or value.get("gather_us") != 5000
        or value.get("measurement_order") != [
            "R2:B32", "R2:B24", "R2:B4", "R2:B1",
            "R0:B32", "R0:B24", "R0:B4", "R0:B1",
        ]
    ):
        raise ProfileError("calibration header changed")
    cut, selection = _validate_selection(path, value)
    capacities = value.get("capacities")
    if (
        type(capacities) is not dict
        or set(capacities) != set(WORKERS)
        or any(not _positive(capacities[name]) or capacities[name] < 32 for name in WORKERS)
    ):
        raise ProfileError("calibration capacity is insufficient")
    expected_ranges = {
        "cuda-prefix": (0, 6),
        "cuda-mid": (6, 8),
        "op12-prefix": (0, cut),
        "op15-mid": (cut, 8),
        "cuda-tail": (8, 48),
    }
    workers = value.get("workers")
    if type(workers) is not dict or set(workers) != set(WORKERS):
        raise ProfileError("calibration worker set changed")
    for name, expected in expected_ranges.items():
        hello = workers[name]
        if (
            type(hello) is not dict
            or (hello.get("layer_start"), hello.get("layer_end")) != expected
            or hello.get("max_streams") != capacities[name]
        ):
            raise ProfileError(f"{name} worker identity changed")
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
        if any(not _positive(row.get(field)) for field in ("wall_us", "max_latency_us", "cuda_work_us")):
            raise ProfileError("calibration timing is invalid")
        tokens = row.get("output_tokens")
        if type(tokens) is not list or len(tokens) != 4 or any(type(token) is not int for token in tokens):
            raise ProfileError("calibration output tokens are invalid")
        events = row.get("events")
        if type(events) is not dict or set(events) != set(WORKERS):
            raise ProfileError("calibration event set is invalid")
        for worker, rows in events.items():
            expected_count = 4 if worker in ACTIVE[route] else 0
            if type(rows) is not list or len(rows) != expected_count:
                raise ProfileError("calibration event count changed")
            if any(
                event.get("status") != "OK"
                or event.get("batch_size") != batch
                or not _positive(event.get("compute_us"))
                for event in rows
            ):
                raise ProfileError("calibration includes a failed or wrong-size batch")
        grouped.setdefault((route, batch), []).append(row)
    for route in ACTIVE:
        for batch in EXPECTED_BATCHES:
            if len(grouped.get((route, batch), [])) != value["reps"]:
                raise ProfileError(f"{route} B{batch} replicate count changed")
    return grouped, cut, selection


def derive_bundle(calibration_path: Path, source_name: str | None = None) -> dict[str, Any]:
    calibration = _load(calibration_path)
    grouped, cut, selection = _validate_calibration(calibration_path, calibration)
    digest = "sha256:" + sha256_file(calibration_path)
    capacities = dict(calibration["capacities"])
    routes = []
    for route_id, batches in ROUTE_BATCHES.items():
        points = []
        for batch in batches:
            rows = grouped[(route_id, batch)]
            duration = max(max(row["wall_us"], row["max_latency_us"]) for row in rows)
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
        "selected_cut": cut,
        "route_layers": {
            "R0": [["cuda-prefix", 0, 6], ["cuda-mid", 6, 8], ["cuda-tail", 8, 48]],
            "R2": [["op12-prefix", 0, cut], ["op15-mid", cut, 8], ["cuda-tail", 8, 48]],
        },
        "source": {"path": source_name or calibration_path.name, "sha256": digest},
        "selection": {
            "path": calibration["selection"]["path"],
            "sha256": calibration["selection"]["sha256"],
            "objective": selection["selected"]["objective"],
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


def load_bundle(path: Path):
    actual = _load(path)
    source = actual.get("source")
    if type(source) is not dict or type(source.get("path")) is not str:
        raise ProfileError("profile source is missing")
    expected = derive_bundle(path.parent / source["path"], source["path"])
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
    return (
        routes,
        dict(actual["capacities"]),
        dict(actual["urgent_reserve"]),
        actual,
        actual["selected_cut"],
    )
