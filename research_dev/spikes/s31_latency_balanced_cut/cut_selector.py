#!/usr/bin/env python3
"""Validate S31 cut measurements and select the least-bottlenecked cut."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "s31-cut-measurement-v1"
SELECTION_SCHEMA = "s31-cut-selection-v1"
WORKERS = ("cuda-prefix", "cuda-mid", "op12-prefix", "op15-mid", "cuda-tail")
ACTIVE = ("op12-prefix", "op15-mid", "cuda-tail")


class CutSelectionError(RuntimeError):
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
            raise CutSelectionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"), object_pairs_hook=_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CutSelectionError(f"cannot load {path}: {exc}") from exc
    if type(value) is not dict:
        raise CutSelectionError(f"{path} must contain an object")
    return value


def nearest_rank(values: Iterable[int], numerator: int, denominator: int) -> int:
    ordered = sorted(values)
    if not ordered:
        raise CutSelectionError("cannot summarize an empty sample")
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[max(0, rank - 1)]


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def validate_measurement(value: dict[str, Any], route_slo_us: int) -> dict[str, Any]:
    if not _positive_int(route_slo_us):
        raise CutSelectionError("route SLO must be a positive integer")
    cut = value.get("candidate", {}).get("cut_layer")
    if (
        value.get("schema") != SCHEMA
        or value.get("status") != "MEASUREMENT_COMPLETE"
        or type(cut) is not int
        or not 1 <= cut < 8
        or value.get("shape") != {
            "input_tokens": 1,
            "output_steps": 4,
            "context": 16,
            "batch_size": 32,
        }
        or value.get("gather_us") != 50000
        or value.get("candidate", {}).get("op12_layers") != [0, cut]
        or value.get("candidate", {}).get("op15_layers") != [cut, 8]
        or value.get("candidate", {}).get("cuda_tail_layers") != [8, 48]
    ):
        raise CutSelectionError("measurement header or candidate identity changed")
    launch = value.get("launch_evidence")
    if (
        type(launch) is not dict
        or launch.get("activation_relay") != "DESKTOP_DIRECT_WIFI"
        or any(
            type(launch.get(field)) is not str
            or len(launch[field]) != 71
            or not launch[field].startswith("sha256:")
            for field in ("phone_session_env", "desktop_session_env")
        )
        or any(
            type(launch.get(field)) is not str
            or len(launch[field]) != 64
            or any(char not in "0123456789abcdef" for char in launch[field])
            for field in (
                "op12_model_sha256",
                "op15_model_sha256",
                "desktop_model_sha256",
            )
        )
        or len({
            launch.get("op12_model_sha256"),
            launch.get("op15_model_sha256"),
            launch.get("desktop_model_sha256"),
        }) != 1
    ):
        raise CutSelectionError("launch evidence is incomplete")
    workers = value.get("workers")
    expected_ranges = {
        "cuda-prefix": (0, 6),
        "cuda-mid": (6, 8),
        "op12-prefix": (0, cut),
        "op15-mid": (cut, 8),
        "cuda-tail": (8, 48),
    }
    if type(workers) is not dict or set(workers) != set(WORKERS):
        raise CutSelectionError("worker set changed")
    for name, expected in expected_ranges.items():
        hello = workers[name]
        if (
            type(hello) is not dict
            or (hello.get("layer_start"), hello.get("layer_end")) != expected
            or not _positive_int(hello.get("max_streams"))
            or hello["max_streams"] < 32
        ):
            raise CutSelectionError(f"{name} HELLO is incompatible")
    reps = value.get("repetitions")
    if type(reps) is not list or len(reps) < 2:
        raise CutSelectionError("at least two repetitions are required")
    if [row.get("rep") for row in reps] != list(range(len(reps))):
        raise CutSelectionError("repetition identifiers are not contiguous")
    stage_samples = {name: [] for name in ACTIVE}
    route_samples = []
    for row in reps:
        if (
            type(row) is not dict
            or not _positive_int(row.get("wall_us"))
            or not _positive_int(row.get("max_latency_us"))
            or row.get("completed_requests") != 32
            or type(row.get("output_tokens")) is not list
            or len(row["output_tokens"]) != 4
            or any(type(token) is not int for token in row["output_tokens"])
        ):
            raise CutSelectionError("repetition outcome is invalid")
        route_samples.append(max(row["wall_us"], row["max_latency_us"]))
        events = row.get("events")
        if type(events) is not dict or set(events) != set(WORKERS):
            raise CutSelectionError("event worker set changed")
        for name in WORKERS:
            rows = events[name]
            expected_count = 4 if name in ACTIVE else 0
            if type(rows) is not list or len(rows) != expected_count:
                raise CutSelectionError(f"{name} event count changed")
            for event in rows:
                if (
                    event.get("status") != "OK"
                    or event.get("batch_size") != 32
                    or not _positive_int(event.get("compute_us"))
                ):
                    raise CutSelectionError(f"{name} contains an invalid event")
                if name in ACTIVE:
                    stage_samples[name].append(event["compute_us"])
    state = value.get("final_software_state")
    if (
        state != {
            "runner_pins": {},
            "software_leases": {name: {} for name in WORKERS},
        }
    ):
        raise CutSelectionError("measurement retained request state")
    op12_p95 = nearest_rank(stage_samples["op12-prefix"], 95, 100)
    op15_p95 = nearest_rank(stage_samples["op15-mid"], 95, 100)
    tail_p95 = nearest_rank(stage_samples["cuda-tail"], 95, 100)
    route_p95 = nearest_rank(route_samples, 95, 100)
    if route_p95 > route_slo_us:
        raise CutSelectionError(f"cut {cut} exceeds the route SLO")
    bottleneck = max(op12_p95, op15_p95)
    return {
        "cut_layer": cut,
        "op12_stage_p95_us": op12_p95,
        "op15_stage_p95_us": op15_p95,
        "cuda_tail_stage_p95_us": tail_p95,
        "phone_bottleneck_p95_us": bottleneck,
        "phone_imbalance_us": abs(op12_p95 - op15_p95),
        "phone_balance_ratio": min(op12_p95, op15_p95) / bottleneck,
        "route_p95_us": route_p95,
        "objective": [
            bottleneck,
            route_p95,
            abs(op12_p95 - op15_p95),
            cut,
        ],
    }


def parse_cuts(value: str) -> tuple[int, ...]:
    try:
        cuts = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cuts must be comma-separated integers") from exc
    if (
        not cuts
        or len(set(cuts)) != len(cuts)
        or any(not 1 <= cut < 8 for cut in cuts)
    ):
        raise argparse.ArgumentTypeError("cuts must be unique integers in [1, 8)")
    return tuple(sorted(cuts))


def build_selection(
    paths: Iterable[Path], route_slo_us: int, expected_cuts: tuple[int, ...],
) -> dict[str, Any]:
    summaries = []
    sources = []
    seen = set()
    for path in paths:
        value = load_json(path)
        summary = validate_measurement(value, route_slo_us)
        cut = summary["cut_layer"]
        if cut in seen:
            raise CutSelectionError(f"duplicate measurement for cut {cut}")
        seen.add(cut)
        summaries.append(summary)
        sources.append({
            "cut_layer": cut,
            "path": path.name,
            "sha256": "sha256:" + sha256_file(path),
        })
    if tuple(sorted(seen)) != tuple(sorted(expected_cuts)):
        raise CutSelectionError("measurement set differs from the declared cuts")
    summaries.sort(key=lambda row: row["cut_layer"])
    sources.sort(key=lambda row: row["cut_layer"])
    selected = min(summaries, key=lambda row: tuple(row["objective"]))
    return {
        "schema": SELECTION_SCHEMA,
        "status": "CUT_SELECTED",
        "route_slo_us": route_slo_us,
        "expected_cuts": list(sorted(expected_cuts)),
        "objective_order": [
            "phone_bottleneck_p95_us",
            "route_p95_us",
            "phone_imbalance_us",
            "cut_layer",
        ],
        "candidates": summaries,
        "sources": sources,
        "selected": selected,
    }


def load_selection(path: Path) -> dict[str, Any]:
    actual = load_json(path)
    sources = actual.get("sources")
    expected_cuts = actual.get("expected_cuts")
    route_slo_us = actual.get("route_slo_us")
    if (
        type(sources) is not list
        or not sources
        or type(expected_cuts) is not list
        or not expected_cuts
        or not _positive_int(route_slo_us)
    ):
        raise CutSelectionError("selection replay metadata is incomplete")
    paths = []
    for source in sources:
        if (
            type(source) is not dict
            or type(source.get("path")) is not str
            or not source["path"]
            or type(source.get("sha256")) is not str
        ):
            raise CutSelectionError("selection source binding is invalid")
        source_path = path.parent / source["path"]
        if source["sha256"] != "sha256:" + sha256_file(source_path):
            raise CutSelectionError("selection source digest mismatch")
        paths.append(source_path)
    if any(type(cut) is not int for cut in expected_cuts):
        raise CutSelectionError("selection cut domain is invalid")
    expected = build_selection(paths, route_slo_us, tuple(expected_cuts))
    if actual != expected or path.read_bytes() != canonical_bytes(expected):
        raise CutSelectionError("selection is not the canonical measurement derivative")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurements", nargs="+", type=Path)
    parser.add_argument("--expected-cuts", type=parse_cuts, required=True)
    parser.add_argument("--route-slo-us", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        result = build_selection(
            args.measurements, args.route_slo_us, args.expected_cuts,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(result))
        print(json.dumps(result["selected"], sort_keys=True))
        return 0
    except BaseException as exc:
        print(json.dumps({
            "schema": SELECTION_SCHEMA,
            "status": "CUT_SELECTION_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, sort_keys=True), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
