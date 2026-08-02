#!/usr/bin/env python3
"""Independently reduce and graph a completed CP0-D desktop campaign."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from PIL import Image, ImageDraw, ImageFont

import validate_inputs


HERE = Path(__file__).resolve().parent
COLORS = {
    "qwen3-8b-q8_0": "#2f6fdd",
    "qwen3-14b-q4_k_m": "#d97706",
    "power": "#18864b",
    "grid": "#d6d9de",
    "text": "#20242a",
    "switch": "#7b2cbf",
}


class AnalysisError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n").encode("ascii")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"{path}: invalid JSON") from exc
    if type(value) is not dict:
        raise AnalysisError(f"{path}: expected object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_bytes().splitlines()):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AnalysisError(f"{path}:{index + 1}: invalid JSON") from exc
        if type(value) is not dict:
            raise AnalysisError(f"{path}:{index + 1}: expected object")
        rows.append(value)
    return rows


def verify_manifest(root: Path) -> None:
    path = root / "SHA256SUMS.txt"
    if not path.is_file():
        raise AnalysisError(f"{root}: missing SHA256SUMS.txt")
    seen: set[str] = set()
    for index, line in enumerate(path.read_text(encoding="ascii").splitlines()):
        fields = line.split("  ", 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise AnalysisError(f"{path}:{index + 1}: malformed")
        digest, name = fields
        if name in seen or name == "SHA256SUMS.txt":
            raise AnalysisError(f"{path}:{index + 1}: duplicate or recursive")
        seen.add(name)
        target = root / name
        if not target.is_file() or digest_file(target) != digest:
            raise AnalysisError(f"{target}: digest mismatch")
    expected = {
        str(item.relative_to(root))
        for item in root.rglob("*")
        if item.is_file() and item.name != "SHA256SUMS.txt"
    }
    if seen != expected:
        raise AnalysisError(f"{root}: manifest file set mismatch")


def parse_stream(path: Path) -> tuple[int, bool]:
    tokens = 0
    final = False
    for index, raw in enumerate(path.read_bytes().splitlines()):
        line = raw.decode("utf-8").strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            row = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"{path}:{index + 1}: invalid stream JSON") from exc
        if type(row) is not dict or "error" in row:
            raise AnalysisError(f"{path}:{index + 1}: stream error")
        values = row.get("tokens", [])
        if type(values) is not list or any(type(value) is not int for value in values):
            raise AnalysisError(f"{path}:{index + 1}: malformed token list")
        tokens += len(values)
        if row.get("stop", False):
            final = True
    return tokens, final


def integrate_power(
        samples: list[dict[str, Any]], start_ns: int, end_ns: int) -> dict[str, int]:
    samples = sorted(samples, key=lambda row: row["t_ns"])
    if not samples or samples[0]["t_ns"] > start_ns or samples[-1]["t_ns"] < end_ns:
        raise AnalysisError("resource samples do not bracket paid window")
    energy_nj = 0
    maximum_gap_ns = 0
    in_window = []
    for row in samples:
        if start_ns <= row["t_ns"] <= end_ns:
            in_window.append(row)
    changes = sum(
        in_window[index]["gpu_power_instant_mw"]
        != in_window[index - 1]["gpu_power_instant_mw"]
        for index in range(1, len(in_window))
    )
    for left, right in zip(samples, samples[1:]):
        lo = max(start_ns, left["t_ns"])
        hi = min(end_ns, right["t_ns"])
        if hi > lo:
            energy_nj += left["gpu_power_instant_mw"] * (hi - lo) // 1_000
            maximum_gap_ns = max(maximum_gap_ns, right["t_ns"] - left["t_ns"])
    if len(in_window) < 20 or changes < 5 or maximum_gap_ns > 1_000_000_000:
        raise AnalysisError("resource sample quality failed")
    return {
        "energy_nj": energy_nj,
        "independent_power_changes": changes,
        "maximum_sample_gap_ns": maximum_gap_ns,
        "sample_count": len(in_window),
        "window_end_ns": end_ns,
        "window_start_ns": start_ns,
    }


def percentile(values: list[int], numerator: int, denominator: int) -> int:
    if not values:
        raise AnalysisError("percentile of empty input")
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[max(1, rank) - 1]


def bracketed_completion_end(
        requests: list[dict[str, Any]],
        samples: list[dict[str, Any]]) -> tuple[int, int]:
    if not requests or not samples:
        raise AnalysisError("missing completion or power samples")
    last_sample_ns = max(row["t_ns"] for row in samples)
    completions = [
        row["completion_ns"] for row in requests
        if row["completion_ns"] <= last_sample_ns
    ]
    if not completions:
        raise AnalysisError("no completion bracketed by power samples")
    return max(completions), len(completions)


def validate_qualification(path: Path, model_id: str) -> dict[str, Any]:
    verify_manifest(path)
    report = read_json(path / "qualification.json")
    if report.get("schema") != "s39-cp0d-qualification-v1" \
            or report.get("status") != "INDEPENDENT_B8_QUALIFICATION_PASS" \
            or report.get("model_id") != model_id \
            or report.get("request_count") != 8:
        raise AnalysisError(f"{path}: qualification identity failed")
    results = report.get("results")
    if type(results) is not list or len(results) != 8:
        raise AnalysisError(f"{path}: qualification results failed")
    for row in results:
        if row.get("predicted_n") != 8 or len(row.get("tokens", [])) != 8:
            raise AnalysisError(f"{path}: qualification token count failed")
        stream_path = path / f"stream-{row['request_index']:03d}.raw"
        token_count, final = parse_stream(stream_path)
        if token_count != 8 or not final:
            raise AnalysisError(f"{stream_path}: stream proof failed")
    ready = report.get("ready")
    if type(ready) is not dict \
            or ready.get("model_id") != model_id \
            or ready.get("gpu", {}).get("gpu_memory_free_bytes", 0) < 536_870_912 \
            or not ready.get("full_cuda_offload_matches"):
        raise AnalysisError(f"{path}: CUDA readiness failed")
    return {
        "b8_elapsed_ns": report["cohort_end_ns"] - report["cohort_start_ns"],
        "load_elapsed_ns": ready["elapsed_ns"],
        "model_id": model_id,
        "post_load_free_vram_bytes": ready["gpu"]["gpu_memory_free_bytes"],
    }


def validate_replay(
        path: Path, expected_regime: str, expected_index: int,
        frozen_requests: list[dict[str, Any]],
        frozen_switches: list[dict[str, Any]]) -> dict[str, Any]:
    verify_manifest(path)
    report = read_json(path / "replay.json")
    if report.get("schema") != "s39-cp0d-replay-v1" \
            or report.get("status") != "RAW_DESKTOP_REPLAY_PASS_ANALYSIS_PENDING" \
            or report.get("regime") != expected_regime \
            or report.get("repeat_index") != expected_index:
        raise AnalysisError(f"{path}: replay identity failed")
    events = read_jsonl(path / "events.jsonl")
    samples = read_jsonl(path / "resource_samples.jsonl")
    starts = [row for row in events if row.get("kind") == "replay_start"]
    ends = [row for row in events if row.get("kind") == "replay_end"]
    if len(starts) != 1 or len(ends) != 1:
        raise AnalysisError(f"{path}: replay bounds failed")
    start_ns = starts[0]["t_ns"]
    end_ns = ends[0]["t_ns"]

    arrival_rows = [
        row for row in events if row.get("kind") == "request_arrival"
    ]
    dispatch_rows = [
        row for row in events if row.get("kind") == "request_dispatched"
    ]
    completion_rows = [
        row for row in events if row.get("kind") == "request_complete"
    ]
    arrivals = {row["request_index"]: row for row in arrival_rows}
    dispatches = {row["request_index"]: row for row in dispatch_rows}
    completions = {row["request_index"]: row for row in completion_rows}
    if len(arrival_rows) != 74 or len(dispatch_rows) != 74 \
            or len(completion_rows) != 74 \
            or len(arrivals) != 74 or len(dispatches) != 74 \
            or len(completions) != 74:
        raise AnalysisError(f"{path}: request conservation failed")

    requests: list[dict[str, Any]] = []
    for frozen in frozen_requests:
        index = frozen["request_index"]
        arrival = arrivals[index]
        dispatch = dispatches[index]
        completion = completions[index]
        scheduled_ns = start_ns + frozen["arrival_us"] * 1000
        if arrival.get("event_id") != frozen["event_id"] \
                or arrival.get("scheduled_t_ns") != scheduled_ns \
                or dispatch.get("model_id") != frozen["model_id"] \
                or completion.get("model_id") != frozen["model_id"] \
                or completion.get("dispatch_model_id") != frozen["model_id"] \
                or completion.get("predicted_n") != 8 \
                or len(completion.get("tokens", [])) != 8:
            raise AnalysisError(f"{path}: request {index} binding failed")
        dispatch_ns = dispatch["t_ns"]
        first_ns = completion["first_token_ns"]
        complete_ns = completion["completion_ns"]
        if not (scheduled_ns <= dispatch_ns <= first_ns <= complete_ns):
            raise AnalysisError(f"{path}: request {index} timing failed")
        stream_tokens, stream_final = parse_stream(
            path / f"stream-{index:03d}.raw"
        )
        if stream_tokens != 8 or not stream_final:
            raise AnalysisError(f"{path}: request {index} stream failed")
        latency_ns = complete_ns - scheduled_ns
        requests.append({
            "completion_latency_ns": latency_ns,
            "completion_ns": complete_ns,
            "event_id": frozen["event_id"],
            "first_token_ns": first_ns,
            "model_id": frozen["model_id"],
            "queue_ns": dispatch_ns - scheduled_ns,
            "request_index": index,
            "service_ttft_ns": first_ns - dispatch_ns,
            "slo_met": latency_ns <= frozen["slo_us"] * 1000,
            "ttft_ns": first_ns - scheduled_ns,
        })

    publications = [
        row for row in events if row.get("kind") == "model_published"
    ]
    if len(publications) != len(frozen_switches):
        raise AnalysisError(f"{path}: switch conservation failed")
    switch_rows: list[dict[str, Any]] = []
    for frozen, row in zip(frozen_switches, publications):
        if row.get("intent_index") != frozen["intent_index"] \
                or row.get("from_model_id") != frozen["from_model_id"] \
                or row.get("to_model_id") != frozen["to_model_id"] \
                or row.get("scheduled_t_ns") != start_ns + frozen["t_us"] * 1000:
            raise AnalysisError(f"{path}: switch binding failed")
        if row["started_ns"] < row["scheduled_t_ns"] \
                or row["published_ns"] < row["started_ns"] \
                or row["publication_gap_ns"] \
                != row["published_ns"] - row["scheduled_t_ns"]:
            raise AnalysisError(f"{path}: switch timing failed")
        switch_rows.append({
            "drain_ns": row["drain_elapsed_ns"],
            "from_model_id": row["from_model_id"],
            "load_ns": row["load"]["elapsed_ns"],
            "publication_gap_ns": row["publication_gap_ns"],
            "published_ns": row["published_ns"],
            "scheduled_ns": row["scheduled_t_ns"],
            "to_model_id": row["to_model_id"],
            "unload_ns": row["unload"]["elapsed_ns"],
        })

    cache_rows = [row for row in events if row.get("kind") == "cache_prepared"]
    if len(cache_rows) != 10:
        raise AnalysisError(f"{path}: expected initial plus nine cache records")
    for row in cache_rows:
        if row.get("regime") != expected_regime:
            raise AnalysisError(f"{path}: cache regime mismatch")
        if expected_regime == "WARM_CACHE":
            if row.get("method") != "VERIFY_PREWARMED_NO_REFILL" \
                    or row.get("resident_ppm", -1) < 950_000:
                raise AnalysisError(f"{path}: warm-cache gate failed")
        else:
            if row.get("method") != "POSIX_FADV_DONTNEED_COMPLETE_FILE" \
                    or row.get("resident_ppm", 1_000_001) > 50_000:
                raise AnalysisError(f"{path}: cold-NVMe gate failed")

    derived_energy = integrate_power(samples, start_ns, end_ns)
    if derived_energy != report.get("energy"):
        raise AnalysisError(f"{path}: energy recomputation mismatch")
    if any(row.get("process_swap_bytes") != 0 for row in samples):
        raise AnalysisError(f"{path}: process swap detected")
    swap_total = {row.get("system_swap_total_bytes") for row in samples}
    if len(swap_total) != 1:
        raise AnalysisError(f"{path}: system swap total changed")
    swap_used = [
        row["system_swap_total_bytes"] - row["system_swap_free_bytes"]
        for row in samples
    ]
    if max(swap_used) > min(swap_used):
        raise AnalysisError(f"{path}: system swap grew")

    paid_ns = end_ns - start_ns
    by_model = Counter(row["model_id"] for row in requests)
    slo_count = sum(row["slo_met"] for row in requests)
    return {
        "completion_latency_p50_ns": percentile(
            [row["completion_latency_ns"] for row in requests], 1, 2
        ),
        "completion_latency_p95_ns": percentile(
            [row["completion_latency_ns"] for row in requests], 95, 100
        ),
        "energy_bracketed_request_count": len(requests),
        "energy_is_lower_bound": False,
        "energy_nj": derived_energy["energy_nj"],
        "energy_per_request_mj": derived_energy["energy_nj"] // 74 // 1_000_000,
        "energy_per_token_mj": derived_energy["energy_nj"] // (74 * 8) // 1_000_000,
        "energy_scope": "SELECTED_GPU_BOARD_COMPLETE_RUN",
        "energy_window_ns": paid_ns,
        "maximum_publication_gap_ns": max(
            row["publication_gap_ns"] for row in switch_rows
        ),
        "mean_load_ns": sum(row["load_ns"] for row in switch_rows) // 9,
        "mean_unload_ns": sum(row["unload_ns"] for row in switch_rows) // 9,
        "model_request_counts": dict(by_model),
        "model_throughput_milli_rps": {
            model_id: count * 1_000_000_000_000 // paid_ns
            for model_id, count in sorted(by_model.items())
        },
        "minimum_system_mem_available_bytes": min(
            row["system_mem_available_bytes"] for row in samples
        ),
        "paid_ns": paid_ns,
        "peak_gpu_memory_used_bytes": max(
            row["gpu_memory_used_bytes"] for row in samples
        ),
        "peak_process_rss_bytes": max(row["process_rss_bytes"] for row in samples),
        "queue_p50_ns": percentile([row["queue_ns"] for row in requests], 1, 2),
        "queue_p95_ns": percentile([row["queue_ns"] for row in requests], 95, 100),
        "regime": expected_regime,
        "repeat_index": expected_index,
        "request_count": len(requests),
        "requests": requests,
        "resource_samples": samples,
        "slo_goodput_milli_rps": slo_count * 1_000_000_000_000 // paid_ns,
        "slo_met_count": slo_count,
        "switches": switch_rows,
        "throughput_milli_rps": 74 * 1_000_000_000_000 // paid_ns,
        "ttft_p50_ns": percentile([row["ttft_ns"] for row in requests], 1, 2),
        "ttft_p95_ns": percentile([row["ttft_ns"] for row in requests], 95, 100),
        "unbracketed_completion_count": 0,
        "verdict": "PASS",
    }


def validate_failed_cold_replay(
        path: Path, expected_index: int,
        frozen_requests: list[dict[str, Any]],
        frozen_switches: list[dict[str, Any]]) -> dict[str, Any]:
    verify_manifest(path)
    if (path / "replay.json").exists():
        raise AnalysisError(f"{path}: failed-cold path has a pass report")
    events = read_jsonl(path / "events.jsonl")
    samples = read_jsonl(path / "resource_samples.jsonl")
    starts = [row for row in events if row.get("kind") == "replay_start"]
    if len(starts) != 1:
        raise AnalysisError(f"{path}: missing replay start")
    start_ns = starts[0]["t_ns"]
    arrival_rows = [
        row for row in events if row.get("kind") == "request_arrival"
    ]
    dispatch_rows = [
        row for row in events if row.get("kind") == "request_dispatched"
    ]
    completion_rows = [
        row for row in events if row.get("kind") == "request_complete"
    ]
    arrivals = {row["request_index"]: row for row in arrival_rows}
    dispatches = {row["request_index"]: row for row in dispatch_rows}
    completions = {row["request_index"]: row for row in completion_rows}
    if len(arrival_rows) != 74 or len(dispatch_rows) != 17 \
            or len(completion_rows) != 17 \
            or len(arrivals) != 74 or len(dispatches) != 17 \
            or len(completions) != 17:
        raise AnalysisError(f"{path}: unexpected failed-cold conservation")
    completed_indices = set(completions)
    expected_completed = {
        row["request_index"] for row in frozen_requests
        if row["model_id"] == "qwen3-14b-q4_k_m"
    }
    expected_stranded = {
        row["request_index"] for row in frozen_requests
        if row["model_id"] == "qwen3-8b-q8_0"
    }
    if completed_indices != expected_completed \
            or set(dispatches) != expected_completed \
            or set(range(74)) - completed_indices != expected_stranded:
        raise AnalysisError(f"{path}: stranded request identity mismatch")
    if set(arrivals) != set(range(74)):
        raise AnalysisError(f"{path}: failed-cold arrival identity mismatch")

    requests: list[dict[str, Any]] = []
    for frozen in frozen_requests:
        index = frozen["request_index"]
        arrival = arrivals[index]
        scheduled_ns = start_ns + frozen["arrival_us"] * 1000
        if arrival.get("event_id") != frozen["event_id"] \
                or arrival.get("scheduled_t_ns") != scheduled_ns:
            raise AnalysisError(f"{path}: failed-cold arrival binding failed")
        if index not in completions:
            continue
        dispatch = dispatches[index]
        completion = completions[index]
        if dispatch.get("model_id") != "qwen3-14b-q4_k_m" \
                or completion.get("model_id") != "qwen3-14b-q4_k_m" \
                or completion.get("predicted_n") != 8 \
                or len(completion.get("tokens", [])) != 8:
            raise AnalysisError(f"{path}: completed request binding failed")
        dispatch_ns = dispatch["t_ns"]
        first_ns = completion["first_token_ns"]
        complete_ns = completion["completion_ns"]
        if not (scheduled_ns <= dispatch_ns <= first_ns <= complete_ns):
            raise AnalysisError(f"{path}: completed request timing failed")
        stream_tokens, stream_final = parse_stream(
            path / f"stream-{index:03d}.raw"
        )
        if stream_tokens != 8 or not stream_final:
            raise AnalysisError(f"{path}: completed stream failed")
        latency_ns = complete_ns - scheduled_ns
        requests.append({
            "completion_latency_ns": latency_ns,
            "completion_ns": complete_ns,
            "event_id": frozen["event_id"],
            "first_token_ns": first_ns,
            "model_id": frozen["model_id"],
            "queue_ns": dispatch_ns - scheduled_ns,
            "request_index": index,
            "service_ttft_ns": first_ns - dispatch_ns,
            "slo_met": latency_ns <= frozen["slo_us"] * 1000,
            "ttft_ns": first_ns - scheduled_ns,
        })

    publications = [
        row for row in events if row.get("kind") == "model_published"
    ]
    if len(publications) != 9:
        raise AnalysisError(f"{path}: failed-cold switch count mismatch")
    switch_rows: list[dict[str, Any]] = []
    for frozen, row in zip(frozen_switches, publications):
        scheduled_ns = start_ns + frozen["t_us"] * 1000
        if row.get("intent_index") != frozen["intent_index"] \
                or row.get("from_model_id") != frozen["from_model_id"] \
                or row.get("to_model_id") != frozen["to_model_id"] \
                or row.get("scheduled_t_ns") != scheduled_ns:
            raise AnalysisError(f"{path}: failed-cold switch binding failed")
        if row["started_ns"] < scheduled_ns \
                or row["published_ns"] < row["started_ns"] \
                or row["publication_gap_ns"] \
                != row["published_ns"] - scheduled_ns:
            raise AnalysisError(f"{path}: failed-cold switch timing failed")
        switch_rows.append({
            "drain_ns": row["drain_elapsed_ns"],
            "from_model_id": row["from_model_id"],
            "load_ns": row["load"]["elapsed_ns"],
            "publication_gap_ns": row["publication_gap_ns"],
            "published_ns": row["published_ns"],
            "scheduled_ns": row["scheduled_t_ns"],
            "to_model_id": row["to_model_id"],
            "unload_ns": row["unload"]["elapsed_ns"],
        })

    cache_rows = [row for row in events if row.get("kind") == "cache_prepared"]
    if len(cache_rows) != 10 or any(
            row.get("regime") != "COLD_NVME"
            or row.get("method") != "POSIX_FADV_DONTNEED_COMPLETE_FILE"
            or row.get("resident_ppm", 1_000_001) > 50_000
            for row in cache_rows):
        raise AnalysisError(f"{path}: failed-cold cache gate failed")
    if any(row.get("process_swap_bytes") != 0 for row in samples):
        raise AnalysisError(f"{path}: failed-cold process swap detected")
    swap_total = {row.get("system_swap_total_bytes") for row in samples}
    if len(swap_total) != 1:
        raise AnalysisError(f"{path}: failed-cold system swap total changed")
    swap_used = [
        row["system_swap_total_bytes"] - row["system_swap_free_bytes"]
        for row in samples
    ]
    if max(swap_used) > min(swap_used):
        raise AnalysisError(f"{path}: failed-cold system swap grew")
    end_ns = max(row["completion_ns"] for row in requests)
    energy_end_ns, bracketed_count = bracketed_completion_end(requests, samples)
    energy = integrate_power(samples, start_ns, energy_end_ns)
    paid_ns = end_ns - start_ns
    slo_count = sum(row["slo_met"] for row in requests)
    return {
        "completion_latency_p50_ns": percentile(
            [row["completion_latency_ns"] for row in requests], 1, 2
        ),
        "completion_latency_p95_ns": percentile(
            [row["completion_latency_ns"] for row in requests], 95, 100
        ),
        "energy_bracketed_request_count": bracketed_count,
        "energy_is_lower_bound": energy_end_ns < end_ns,
        "energy_nj": energy["energy_nj"],
        "energy_per_request_mj": energy["energy_nj"] // 17 // 1_000_000,
        "energy_per_token_mj": energy["energy_nj"] // (17 * 8) // 1_000_000,
        "energy_scope": "SELECTED_GPU_BOARD_BRACKETED_PREFIX",
        "energy_window_ns": energy_end_ns - start_ns,
        "maximum_publication_gap_ns": max(
            row["publication_gap_ns"] for row in switch_rows
        ),
        "mean_load_ns": sum(row["load_ns"] for row in switch_rows) // 9,
        "mean_unload_ns": sum(row["unload_ns"] for row in switch_rows) // 9,
        "model_request_counts": {
            "qwen3-14b-q4_k_m": 17,
            "qwen3-8b-q8_0": 0,
        },
        "model_throughput_milli_rps": {
            "qwen3-14b-q4_k_m": 17 * 1_000_000_000_000 // paid_ns,
            "qwen3-8b-q8_0": 0,
        },
        "minimum_system_mem_available_bytes": min(
            row["system_mem_available_bytes"] for row in samples
        ),
        "paid_ns": paid_ns,
        "peak_gpu_memory_used_bytes": max(
            row["gpu_memory_used_bytes"] for row in samples
        ),
        "peak_process_rss_bytes": max(row["process_rss_bytes"] for row in samples),
        "queue_p50_ns": percentile([row["queue_ns"] for row in requests], 1, 2),
        "queue_p95_ns": percentile([row["queue_ns"] for row in requests], 95, 100),
        "regime": "COLD_NVME",
        "repeat_index": expected_index,
        "request_count": 17,
        "requests": requests,
        "resource_samples": samples,
        "slo_goodput_milli_rps": slo_count * 1_000_000_000_000 // paid_ns,
        "slo_met_count": slo_count,
        "stranded_request_count": 57,
        "switches": switch_rows,
        "throughput_milli_rps": 17 * 1_000_000_000_000 // paid_ns,
        "ttft_p50_ns": percentile([row["ttft_ns"] for row in requests], 1, 2),
        "ttft_p95_ns": percentile([row["ttft_ns"] for row in requests], 95, 100),
        "unbracketed_completion_count": (
            len(requests) - bracketed_count
        ),
        "verdict": "FAIL_STRANDED_DOMINANT_MODEL_QUEUE",
    }


def median_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (row["paid_ns"], row["repeat_index"]))
    return ordered[len(ordered) // 2]


def trailing_throughput(counts: list[int], window_s: int) -> list[float]:
    if window_s <= 0:
        raise AnalysisError("throughput window must be positive")
    return [
        sum(counts[max(0, index - window_s + 1):index + 1]) / window_s
        for index in range(len(counts))
    ]


def cumulative_energy_j(
        samples: list[dict[str, Any]], start_ns: int, end_ns: int,
        duration_s: int) -> list[float]:
    ordered = sorted(samples, key=lambda row: row["t_ns"])
    if not ordered or ordered[0]["t_ns"] > start_ns \
            or ordered[-1]["t_ns"] < end_ns:
        raise AnalysisError("resource samples do not bracket energy series")
    result: list[float] = []
    for second in range(1, duration_s + 1):
        prefix_end_ns = min(end_ns, start_ns + second * 1_000_000_000)
        energy_nj = 0
        for left, right in zip(ordered, ordered[1:]):
            lo = max(start_ns, left["t_ns"])
            hi = min(prefix_end_ns, right["t_ns"])
            if hi > lo:
                energy_nj += (
                    left["gpu_power_instant_mw"] * (hi - lo) // 1_000
                )
        result.append(energy_nj / 1_000_000_000)
    return result


def bin_run(run: dict[str, Any]) -> dict[str, Any]:
    start_ns = min(
        min(row["completion_ns"] - row["completion_latency_ns"]
            for row in run["requests"]),
        run["switches"][0]["scheduled_ns"],
    )
    # The first scheduled arrival is offset from paid start. Recover paid start
    # from the energy window, which is independently recomputed.
    paid_start = min(row["t_ns"] for row in run["resource_samples"])
    energy_start = run["resource_samples"][0]["t_ns"]
    del start_ns, energy_start
    paid_start = run["resource_samples"][0]["t_ns"]
    # Use the switch schedule to recover the exact paid origin. The first frozen
    # switch is at 3 seconds.
    paid_start = run["switches"][0]["scheduled_ns"] - 3_000_000_000
    duration_s = math.ceil(run["paid_ns"] / 1_000_000_000)
    completions = {
        model: [0] * max(1, duration_s)
        for model in ("qwen3-8b-q8_0", "qwen3-14b-q4_k_m")
    }
    for row in run["requests"]:
        index = min(
            len(completions[row["model_id"]]) - 1,
            max(0, (row["completion_ns"] - paid_start) // 1_000_000_000),
        )
        completions[row["model_id"]][index] += 1
    power_sum = [0] * max(1, duration_s)
    power_count = [0] * max(1, duration_s)
    for row in run["resource_samples"]:
        index = (row["t_ns"] - paid_start) // 1_000_000_000
        if 0 <= index < duration_s:
            power_sum[index] += row["gpu_power_instant_mw"]
            power_count[index] += 1
    power_w = [
        0.0 if count == 0 else power_sum[index] / count / 1000.0
        for index, count in enumerate(power_count)
    ]
    switches_s = [
        (row["published_ns"] - paid_start) / 1e9
        for row in run["switches"]
    ]
    throughput_window_s = 5
    energy_end_ns = paid_start + run["energy_window_ns"]
    cumulative_energy = cumulative_energy_j(
        run["resource_samples"], paid_start, energy_end_ns, duration_s
    )
    if abs(cumulative_energy[-1] - run["energy_nj"] / 1e9) > 1e-6:
        raise AnalysisError("energy timeline does not match reduced energy")
    return {
        "completions": completions,
        "cumulative_energy_j": cumulative_energy,
        "duration_s": duration_s,
        "energy_is_lower_bound": run["energy_is_lower_bound"],
        "model_throughput_rps": {
            model_id: value / 1000.0
            for model_id, value in run["model_throughput_milli_rps"].items()
        },
        "power_w": power_w,
        "request_count": run["request_count"],
        "slo_met_count": run["slo_met_count"],
        "stranded_request_count": run.get("stranded_request_count", 0),
        "switches_s": switches_s,
        "throughput_series_rps": {
            model_id: trailing_throughput(values, throughput_window_s)
            for model_id, values in completions.items()
        },
        "throughput_window_s": throughput_window_s,
        "throughput_rps": run["throughput_milli_rps"] / 1000.0,
    }


def make_svg(warm: dict[str, Any], cold: dict[str, Any], path: Path) -> None:
    width, height = 1400, 850
    left, right, top, bottom = 90, 80, 140, 70
    panel_h = 235
    gap = 125
    max_duration = max(warm["duration_s"], cold["duration_s"])
    max_throughput = max(
        1.0,
        max(max(values) for run in (warm, cold)
            for values in run["throughput_series_rps"].values()),
    )
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="70" y="35" font-family="sans-serif" font-size="24" '
        'font-weight="700">RTX 4060 Ti model throughput over time</text>',
        '<text x="70" y="58" font-family="sans-serif" font-size="14" '
        'fill="#555">Five-second trailing throughput with model publication points</text>',
    ]
    for panel_index, (name, run) in enumerate(
            (("Warm page cache", warm), ("Cold NVMe", cold))):
        y0 = top + panel_index * (panel_h + gap)
        x0, x1 = left, width - right
        y1 = y0 + panel_h
        body.append(
            f'<text x="{x0}" y="{y0 - 38}" font-family="sans-serif" '
            f'font-size="18" font-weight="700">{name}</text>'
        )
        result_text = (
            f'{run["request_count"]}/74 completed, '
            f'{run["slo_met_count"]} within SLO'
        )
        if run["stranded_request_count"]:
            result_text += f', {run["stranded_request_count"]} stranded'
        body.append(
            f'<text x="{x0 + 190}" y="{y0 - 38}" font-family="sans-serif" '
            f'font-size="14" fill="#555">{result_text}</text>'
        )
        rates = run["model_throughput_rps"]
        throughput_text = (
            f'Average throughput: Qwen3-8B {rates["qwen3-8b-q8_0"]:.3f} req/s'
            f' | Qwen3-14B {rates["qwen3-14b-q4_k_m"]:.3f} req/s'
            f' | total {run["throughput_rps"]:.3f} req/s'
        )
        body.append(
            f'<text x="{x0}" y="{y0 - 13}" font-family="sans-serif" '
            f'font-size="14" fill="#333">{throughput_text}</text>'
        )
        for tick in range(6):
            y = y1 - tick * panel_h / 5
            value = tick * max_throughput / 5
            body.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                f'stroke="{COLORS["grid"]}"/>'
            )
            body.append(
                f'<text x="{x0 - 10}" y="{y + 5:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">{value:.1f}</text>'
            )
        for second in range(0, max_duration + 1, max(1, max_duration // 8)):
            x = x0 + (x1 - x0) * second / max_duration
            body.append(
                f'<text x="{x:.1f}" y="{y1 + 24}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12">{second}</text>'
            )
        for switch in run["switches_s"]:
            x = x0 + (x1 - x0) * switch / max_duration
            body.append(
                f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" '
                f'stroke="{COLORS["switch"]}" stroke-width="1.5" stroke-dasharray="5 4"/>'
            )
        for model, values in run["throughput_series_rps"].items():
            points = []
            for index, value in enumerate(values):
                x = x0 + (x1 - x0) * (index + 0.5) / max_duration
                y = y1 - panel_h * value / max_throughput
                points.append(f"{x:.1f},{y:.1f}")
            body.append(
                f'<polyline points="{" ".join(points)}" fill="none" '
                f'stroke="{COLORS[model]}" stroke-width="2.5"/>'
            )
        body.append(
            f'<text x="{x0 - 58}" y="{y0 + panel_h / 2:.1f}" '
            'transform="rotate(-90 {x} {y})" font-family="sans-serif" '
            'font-size="12">5s trailing throughput (req/s)</text>'.format(
                x=x0 - 58, y=y0 + panel_h / 2
            )
        )
    body.append(
        f'<text x="{width / 2:.1f}" y="{height - 54}" text-anchor="middle" '
        'font-family="sans-serif" font-size="13">time since replay start (s)</text>'
    )
    legend_y = height - 26
    legend = [
        ("Qwen3-8B throughput", COLORS["qwen3-8b-q8_0"]),
        ("Qwen3-14B throughput", COLORS["qwen3-14b-q4_k_m"]),
        ("model published", COLORS["switch"]),
    ]
    x = 160
    for label, color in legend:
        body.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 24}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        body.append(
            f'<text x="{x + 30}" y="{legend_y + 5}" font-family="sans-serif" '
            f'font-size="13">{label}</text>'
        )
        x += 350
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="ascii")


def make_png(warm: dict[str, Any], cold: dict[str, Any], path: Path) -> None:
    width, height = 1400, 850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    title = ImageFont.load_default(size=26)
    draw.text((70, 12), "RTX 4060 Ti model throughput over time",
              fill=COLORS["text"], font=title)
    draw.text((70, 50),
              "Five-second trailing throughput with model publication points",
              fill="#555555", font=font)
    left, right, top, panel_h, gap = 90, 80, 140, 235, 125
    max_duration = max(warm["duration_s"], cold["duration_s"])
    max_throughput = max(
        1.0,
        max(max(values) for run in (warm, cold)
            for values in run["throughput_series_rps"].values()),
    )
    for panel_index, (name, run) in enumerate(
            (("Warm page cache", warm), ("Cold NVMe", cold))):
        y0 = top + panel_index * (panel_h + gap)
        y1 = y0 + panel_h
        x0, x1 = left, width - right
        draw.text((x0, y0 - 48), name, fill=COLORS["text"], font=font)
        result_text = (
            f'{run["request_count"]}/74 completed, '
            f'{run["slo_met_count"]} within SLO'
        )
        if run["stranded_request_count"]:
            result_text += f', {run["stranded_request_count"]} stranded'
        draw.text((x0 + 190, y0 - 48), result_text,
                  fill="#555555", font=font)
        rates = run["model_throughput_rps"]
        throughput_text = (
            f'Average throughput: Qwen3-8B {rates["qwen3-8b-q8_0"]:.3f} req/s'
            f' | Qwen3-14B {rates["qwen3-14b-q4_k_m"]:.3f} req/s'
            f' | total {run["throughput_rps"]:.3f} req/s'
        )
        draw.text((x0, y0 - 24), throughput_text,
                  fill="#333333", font=font)
        for tick in range(6):
            y = int(y1 - tick * panel_h / 5)
            draw.line((x0, y, x1, y), fill=COLORS["grid"], width=1)
            draw.text((x0 - 55, y - 8),
                      f"{tick * max_throughput / 5:.1f}",
                      fill=COLORS["text"], font=font)
        for switch in run["switches_s"]:
            x = int(x0 + (x1 - x0) * switch / max_duration)
            for y in range(y0, y1, 10):
                draw.line((x, y, x, min(y + 5, y1)),
                          fill=COLORS["switch"], width=2)
        for model, values in run["throughput_series_rps"].items():
            points = [
                (
                    int(x0 + (x1 - x0) * (index + 0.5) / max_duration),
                    int(y1 - panel_h * value / max_throughput),
                )
                for index, value in enumerate(values)
            ]
            if len(points) >= 2:
                draw.line(points, fill=COLORS[model], width=3)
        for second in range(0, max_duration + 1, max(1, max_duration // 8)):
            x = int(x0 + (x1 - x0) * second / max_duration)
            draw.text((x - 12, y1 + 8), str(second),
                      fill=COLORS["text"], font=font)
    draw.text((width // 2 - 85, height - 58),
              "time since replay start (s)",
              fill=COLORS["text"], font=font)
    legend_y = height - 28
    legend = [
        ("Qwen3-8B throughput", COLORS["qwen3-8b-q8_0"]),
        ("Qwen3-14B throughput", COLORS["qwen3-14b-q4_k_m"]),
        ("model published", COLORS["switch"]),
    ]
    x = 190
    for label, color in legend:
        draw.line((x, legend_y, x + 24, legend_y), fill=color, width=4)
        draw.text((x + 30, legend_y - 8), label, fill=COLORS["text"], font=font)
        x += 350
    image.save(path)


def make_energy_svg(
        warm: dict[str, Any], cold: dict[str, Any], path: Path) -> None:
    width, height = 1400, 850
    left, right, top = 90, 80, 140
    panel_h, gap = 235, 125
    max_duration = max(warm["duration_s"], cold["duration_s"])
    max_energy = max(
        1.0, warm["cumulative_energy_j"][-1],
        cold["cumulative_energy_j"][-1],
    )
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="70" y="35" font-family="sans-serif" font-size="24" '
        'font-weight="700">RTX 4060 Ti selected-GPU energy over time</text>',
        '<text x="70" y="58" font-family="sans-serif" font-size="14" '
        'fill="#555">Cumulative GPU board energy with model publication points</text>',
    ]
    for panel_index, (name, run) in enumerate(
            (("Warm page cache", warm), ("Cold NVMe", cold))):
        y0 = top + panel_index * (panel_h + gap)
        y1 = y0 + panel_h
        x0, x1 = left, width - right
        qualifier = ">=" if run["energy_is_lower_bound"] else ""
        body.append(
            f'<text x="{x0}" y="{y0 - 38}" font-family="sans-serif" '
            f'font-size="18" font-weight="700">{name}</text>'
        )
        body.append(
            f'<text x="{x0 + 190}" y="{y0 - 38}" font-family="sans-serif" '
            f'font-size="14" fill="#555">final measured energy '
            f'{qualifier}{run["cumulative_energy_j"][-1]:.1f} J</text>'
        )
        scope = (
            "bracketed prefix; failed run lower bound"
            if run["energy_is_lower_bound"]
            else "complete paid replay"
        )
        body.append(
            f'<text x="{x0}" y="{y0 - 13}" font-family="sans-serif" '
            f'font-size="14" fill="#333">{scope}</text>'
        )
        for tick in range(6):
            y = y1 - tick * panel_h / 5
            value = tick * max_energy / 5
            body.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                f'stroke="{COLORS["grid"]}"/>'
            )
            body.append(
                f'<text x="{x0 - 10}" y="{y + 5:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">{value:.0f}</text>'
            )
        for second in range(0, max_duration + 1, max(1, max_duration // 8)):
            x = x0 + (x1 - x0) * second / max_duration
            body.append(
                f'<text x="{x:.1f}" y="{y1 + 24}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12">{second}</text>'
            )
        for switch in run["switches_s"]:
            x = x0 + (x1 - x0) * switch / max_duration
            body.append(
                f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" '
                f'stroke="{COLORS["switch"]}" stroke-width="1.5" '
                'stroke-dasharray="5 4"/>'
            )
        points = [f"{x0:.1f},{y1:.1f}"]
        for index, value in enumerate(run["cumulative_energy_j"]):
            x = x0 + (x1 - x0) * (index + 1) / max_duration
            y = y1 - panel_h * value / max_energy
            points.append(f"{x:.1f},{y:.1f}")
        body.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{COLORS["power"]}" stroke-width="3"/>'
        )
        body.append(
            f'<text x="{x0 - 58}" y="{y0 + panel_h / 2:.1f}" '
            'transform="rotate(-90 {x} {y})" font-family="sans-serif" '
            'font-size="12">cumulative GPU energy (J)</text>'.format(
                x=x0 - 58, y=y0 + panel_h / 2
            )
        )
    body.append(
        f'<text x="{width / 2:.1f}" y="{height - 54}" text-anchor="middle" '
        'font-family="sans-serif" font-size="13">time since replay start (s)</text>'
    )
    legend_y = height - 26
    for index, (label, color) in enumerate((
            ("cumulative selected-GPU energy", COLORS["power"]),
            ("model published", COLORS["switch"]))):
        x = 330 + index * 470
        body.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 24}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        body.append(
            f'<text x="{x + 30}" y="{legend_y + 5}" font-family="sans-serif" '
            f'font-size="13">{label}</text>'
        )
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="ascii")


def make_energy_png(
        warm: dict[str, Any], cold: dict[str, Any], path: Path) -> None:
    width, height = 1400, 850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    title = ImageFont.load_default(size=26)
    draw.text((70, 12), "RTX 4060 Ti selected-GPU energy over time",
              fill=COLORS["text"], font=title)
    draw.text((70, 50),
              "Cumulative GPU board energy with model publication points",
              fill="#555555", font=font)
    left, right, top, panel_h, gap = 90, 80, 140, 235, 125
    max_duration = max(warm["duration_s"], cold["duration_s"])
    max_energy = max(
        1.0, warm["cumulative_energy_j"][-1],
        cold["cumulative_energy_j"][-1],
    )
    for panel_index, (name, run) in enumerate(
            (("Warm page cache", warm), ("Cold NVMe", cold))):
        y0 = top + panel_index * (panel_h + gap)
        y1 = y0 + panel_h
        x0, x1 = left, width - right
        qualifier = ">=" if run["energy_is_lower_bound"] else ""
        draw.text((x0, y0 - 48), name, fill=COLORS["text"], font=font)
        draw.text(
            (x0 + 190, y0 - 48),
            f'final measured energy {qualifier}'
            f'{run["cumulative_energy_j"][-1]:.1f} J',
            fill="#555555", font=font,
        )
        scope = (
            "bracketed prefix; failed run lower bound"
            if run["energy_is_lower_bound"]
            else "complete paid replay"
        )
        draw.text((x0, y0 - 24), scope, fill="#333333", font=font)
        for tick in range(6):
            y = int(y1 - tick * panel_h / 5)
            draw.line((x0, y, x1, y), fill=COLORS["grid"], width=1)
            draw.text(
                (x0 - 62, y - 8), f"{tick * max_energy / 5:.0f}",
                fill=COLORS["text"], font=font,
            )
        for switch in run["switches_s"]:
            x = int(x0 + (x1 - x0) * switch / max_duration)
            for y in range(y0, y1, 10):
                draw.line(
                    (x, y, x, min(y + 5, y1)),
                    fill=COLORS["switch"], width=2,
                )
        points = [(x0, y1)]
        points.extend(
            (
                int(x0 + (x1 - x0) * (index + 1) / max_duration),
                int(y1 - panel_h * value / max_energy),
            )
            for index, value in enumerate(run["cumulative_energy_j"])
        )
        if len(points) >= 2:
            draw.line(points, fill=COLORS["power"], width=3)
        for second in range(0, max_duration + 1, max(1, max_duration // 8)):
            x = int(x0 + (x1 - x0) * second / max_duration)
            draw.text(
                (x - 12, y1 + 8), str(second),
                fill=COLORS["text"], font=font,
            )
    draw.text((width // 2 - 85, height - 58),
              "time since replay start (s)",
              fill=COLORS["text"], font=font)
    legend_y = height - 28
    for index, (label, color) in enumerate((
            ("cumulative selected-GPU energy", COLORS["power"]),
            ("model published", COLORS["switch"]))):
        x = 330 + index * 470
        draw.line((x, legend_y, x + 24, legend_y), fill=color, width=4)
        draw.text(
            (x + 30, legend_y - 8), label,
            fill=COLORS["text"], font=font,
        )
    image.save(path)


def public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in run.items()
        if key not in {"requests", "resource_samples", "switches"}
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_inputs.validate(HERE)
    campaign = args.campaign.resolve()
    if args.output.exists():
        raise AnalysisError(f"output exists: {args.output}")
    args.output.mkdir(parents=True)

    frozen_requests = read_jsonl(HERE / "DESKTOP_REQUESTS.jsonl")
    frozen_switches = read_jsonl(HERE / "DESKTOP_SWITCHES.jsonl")
    qualification = [
        validate_qualification(
            campaign / "qualification-qwen3-14b", "qwen3-14b-q4_k_m"
        ),
        validate_qualification(
            campaign / "qualification-qwen3-8b", "qwen3-8b-q8_0"
        ),
    ]
    nonco_path = campaign / "noncoresidency"
    verify_manifest(nonco_path)
    nonco = read_json(nonco_path / "non_coresidency.json")
    if nonco.get("status") != "PHYSICAL_NON_CORESIDENCY_PASS" \
            or len(nonco.get("attempts", [])) != 2 \
            or any(row.get("second_eligible") is not False
                   for row in nonco["attempts"]):
        raise AnalysisError("non-co-residency proof failed")

    contract = read_json(HERE / "DESKTOP_BASELINE_CONTRACT.json")
    runs: list[dict[str, Any]] = []
    for index, regime in enumerate(contract["replay"]["run_order"]):
        path = campaign / f"replay-{index:02d}-{regime.lower()}"
        if regime == "COLD_NVME" and not (path / "replay.json").exists():
            runs.append(validate_failed_cold_replay(
                path, index, frozen_requests, frozen_switches
            ))
        else:
            runs.append(validate_replay(
                path, regime, index, frozen_requests, frozen_switches
            ))
    warm_runs = [row for row in runs if row["regime"] == "WARM_CACHE"]
    cold_runs = [row for row in runs if row["regime"] == "COLD_NVME"]
    representative_warm = median_run(warm_runs)
    representative_cold = median_run(cold_runs)
    warm_bins = bin_run(representative_warm)
    cold_bins = bin_run(representative_cold)
    (args.output / "timeline_data.json").write_bytes(canonical({
        "COLD_NVME": cold_bins,
        "WARM_CACHE": warm_bins,
        "schema": "s39-cp0d-timeline-data-v1",
    }))
    make_svg(
        warm_bins, cold_bins, args.output / "throughput_timeline.svg"
    )
    make_png(
        warm_bins, cold_bins, args.output / "throughput_timeline.png"
    )
    make_energy_svg(
        warm_bins, cold_bins, args.output / "energy_timeline.svg"
    )
    make_energy_png(
        warm_bins, cold_bins, args.output / "energy_timeline.png"
    )

    summary = {
        "campaign_path": str(campaign),
        "contract_sha256": digest_file(HERE / "DESKTOP_BASELINE_CONTRACT.json"),
        "non_coresidency": {
            "attempt_count": 2,
            "status": nonco["status"],
        },
        "qualification": qualification,
        "representative_repeats": {
            "COLD_NVME": representative_cold["repeat_index"],
            "WARM_CACHE": representative_warm["repeat_index"],
        },
        "runs": [public_run(row) for row in runs],
        "schema": "s39-cp0d-analysis-v1",
        "status": (
            "DESKTOP_BASELINE_WARM_PASS_COLD_FAIL"
            if all(row["verdict"] == "PASS" for row in warm_runs)
            and all(row["verdict"] == "FAIL_STRANDED_DOMINANT_MODEL_QUEUE"
                    for row in cold_runs)
            else "DESKTOP_BASELINE_UNEXPECTED_RESULT"
        ),
    }
    (args.output / "summary.json").write_bytes(canonical(summary))
    files = sorted(
        item for item in args.output.iterdir()
        if item.is_file() and item.name != "SHA256SUMS.txt"
    )
    (args.output / "SHA256SUMS.txt").write_text(
        "".join(f"{digest_file(item)}  {item.name}\n" for item in files),
        encoding="ascii",
    )
    print(json.dumps({
        "cold_repeat": representative_cold["repeat_index"],
        "status": summary["status"],
        "warm_repeat": representative_warm["repeat_index"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"CP0D_ANALYSIS_ERROR: {exc}")
        raise SystemExit(2)
