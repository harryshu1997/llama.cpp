#!/usr/bin/env python3
"""Bind measured batch evidence to a live S14 route."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from power_frontier_policy import CertifiedBatchPoint
from priority_batch_runtime import RouteConfig


class ProfileAdapterError(ValueError):
    pass


def _load(path: Path) -> dict:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProfileAdapterError(f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=no_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileAdapterError(str(exc)) from exc
    if type(value) is not dict:
        raise ProfileAdapterError("profile result must be an object")
    return value


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _hex_digest(name: str, value: object) -> str:
    if type(value) is not str or len(value) != 64 \
            or any(char not in "0123456789abcdef" for char in value):
        raise ProfileAdapterError(f"invalid {name}")
    return value


def _eligible_row(data: dict) -> dict:
    if data.get("schema") != "s14-stage-b-deep-head-placement-v1" \
            or data.get("status") != "HEAD_SWEEP_COMPLETE":
        raise ProfileAdapterError("batch result is not a completed passing sweep")
    device = data.get("device")
    if type(device) is not dict or device.get("serial") != "3C15AU002CL00000" \
            or device.get("backend") != "HTP0":
        raise ProfileAdapterError("batch result has the wrong device or backend")
    thermal = data.get("thermal")
    if type(thermal) is not dict:
        raise ProfileAdapterError("batch result has no thermal evidence")
    for name, limit_name in (("start", "start_max_millic"), ("end", "end_max_millic")):
        snapshot = thermal.get(name)
        limit = thermal.get(limit_name)
        if type(snapshot) is not dict or type(limit) is not int \
                or snapshot.get("valid") is not True:
            raise ProfileAdapterError("batch result has invalid thermal evidence")
        sensors = snapshot.get("sensors_millic")
        maximum = snapshot.get("max_millic")
        if type(sensors) is not dict or not sensors or type(maximum) is not int \
                or maximum != max(sensors.values()) or maximum > limit:
            raise ProfileAdapterError("batch result exceeds its thermal envelope")
    batch = data.get("batch")
    requests = data.get("requests")
    if type(batch) is not int or batch <= 0 or type(requests) is not int \
            or requests < batch or requests % batch != 0:
        raise ProfileAdapterError("batch result does not contain complete native batches")
    rows = data.get("per_k")
    if type(rows) is not list or len(rows) != 1 or type(rows[0]) is not dict:
        raise ProfileAdapterError("batch result must contain one island row")
    row = rows[0]
    if row.get("layer_range") != [0, 8] or row.get("batch") != batch \
            or row.get("n_requests_measured") != requests \
            or row.get("placement_status") != "SCHEDULED_PLACEMENT_OK" \
            or row.get("missing_buffer_compute_nodes") != 0 \
            or row.get("host_returncode") != 0 \
            or row.get("token_match_vs_mono") is not True \
            or row.get("all_tokens_match_vs_mono") is not True:
        raise ProfileAdapterError("batch row failed range, completion, placement, or correctness")
    cert = row.get("placement_cert")
    if type(cert) is not dict or cert.get("status") != "SCHEDULED_PLACEMENT_OK" \
            or cert.get("layer_start") != 0 or cert.get("layer_end") != 8:
        raise ProfileAdapterError("placement certificate is missing or has the wrong range")
    htp_nodes = 0
    for op, buffers in cert.get("compute_by_op_and_buffer", {}).items():
        if type(buffers) is not dict:
            raise ProfileAdapterError("invalid placement map")
        for buffer, count in buffers.items():
            if type(count) is not int or count <= 0:
                raise ProfileAdapterError("invalid placement count")
            if "HTP" in buffer:
                htp_nodes += count
            elif op != "GET_ROWS":
                raise ProfileAdapterError(f"undeclared CPU work {op}@{buffer}")
    if htp_nodes <= 0:
        raise ProfileAdapterError("placement certificate has no HTP work")
    duration = row.get("request_wall_us_p50")
    if type(duration) is not int or duration <= 0:
        raise ProfileAdapterError("batch row has no positive route duration")
    _hex_digest("phone binary digest", data.get("phone_binary_sha256"))
    _hex_digest("shard digest", row.get("shard_sha256"))
    return row


def load_op15_head_route(paths: list[Path], route_epoch: int) -> RouteConfig:
    if type(route_epoch) is not int or route_epoch <= 0 or not paths:
        raise ProfileAdapterError("route epoch and profile paths are required")
    rows_by_batch: dict[int, list[tuple[dict, str]]] = {}
    bindings = []
    seen_digests = set()
    for path in paths:
        data = _load(path)
        row = _eligible_row(data)
        batch = data["batch"]
        digest = _digest(path)
        if digest in seen_digests:
            raise ProfileAdapterError("duplicate profile artifact content")
        seen_digests.add(digest)
        bindings.append(f"{batch}:{digest}")
        rows_by_batch.setdefault(batch, []).append((row, digest))
    points = []
    for batch, records in sorted(rows_by_batch.items()):
        if len(records) < 7:
            raise ProfileAdapterError(f"batch {batch} has fewer than seven independent processes")
        durations = sorted(row["request_wall_us_p50"] for row, _ in records)
        mean = statistics.fmean(durations)
        if statistics.pstdev(durations) / mean > 0.05:
            raise ProfileAdapterError(f"batch {batch} exceeds the 0.05 process CoV gate")
        batch_binding = "\n".join(sorted(digest for _, digest in records))
        batch_digest = "sha256:" + hashlib.sha256(batch_binding.encode()).hexdigest()
        points.append(CertifiedBatchPoint(
            batch,
            durations[len(durations) // 2],
            f"{batch_digest}#same-batch-token-correctness-7proc",
            f"{batch_digest}#scheduled-placement-7proc",
        ))
    if 1 not in rows_by_batch:
        raise ProfileAdapterError("route profile must include batch 1")
    profile_id = "sha256:" + hashlib.sha256("\n".join(sorted(bindings)).encode()).hexdigest()
    return RouteConfig(
        route_id="op15-gemma-head-0-8",
        service_class="generation",
        model_id="gemma-4-12b-it-f16",
        island_id="gemma-head-0-8",
        profile_id=profile_id,
        route_epoch=route_epoch,
        roofline_class="memory_bound",
        points=tuple(sorted(points, key=lambda point: point.batch_size)),
        high_priority_max=0,
    )
