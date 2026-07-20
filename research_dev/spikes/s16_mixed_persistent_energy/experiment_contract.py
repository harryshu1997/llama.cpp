#!/usr/bin/env python3
"""Pure validation and aggregation for the S16 physical acquisition."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


BATCH = 32
N_GEN = 8
LOW_COHORTS = 20
LOW_SLO_US = 5_000_000
HIGH_P95_RATIO_MAX = 1.05
MIN_UPDATES = 100
MAX_GAP_US = 250_000
NVML_UNCERTAINTY_MW = 5_000
FULL_SCHEDULE = (("P0", 0), ("P2", 0), ("P2", 1),
                 ("P0", 1), ("P0", 2), ("P2", 2))


class ContractError(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            size += len(chunk)
            value.update(chunk)
    if size == 0:
        raise ContractError(f"empty artifact: {path}")
    return "sha256:" + value.hexdigest()


def strict_object(payload: bytes, label: str, canonical_required: bool = True) -> dict[str, Any]:
    def no_duplicates(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ContractError(f"duplicate key in {label}: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(payload, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid {label}: {exc}") from exc
    if type(value) is not dict:
        raise ContractError(f"{label} is not an object")
    if canonical_required and canonical(value) != payload:
        raise ContractError(f"{label} is not canonical")
    return value


def percentile(values: list[int], numerator: int, denominator: int) -> int:
    if not values or numerator < 0 or numerator > denominator or denominator <= 0:
        raise ContractError("invalid percentile input")
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[max(0, rank - 1)]


def parse_power_jsonl(payload: bytes) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise ContractError("power artifact is empty or unterminated")
    rows = []
    previous = -1
    for index, line in enumerate(payload.splitlines(keepends=True)):
        row = strict_object(line, f"power row {index}")
        expected = {"t_us", "power_mw", "power_limit_mw", "util_milli_pct", "pstate"}
        if set(row) != expected:
            raise ContractError("power row has missing or unknown fields")
        for key in ("t_us", "power_mw", "power_limit_mw", "util_milli_pct"):
            if type(row[key]) is not int or row[key] < 0:
                raise ContractError(f"power row has invalid {key}")
        if type(row["pstate"]) is not str or not row["pstate"]:
            raise ContractError("power row has invalid pstate")
        if row["t_us"] <= previous:
            raise ContractError("power timestamps are not strictly increasing")
        previous = row["t_us"]
        rows.append(row)
    if len(rows) < 2:
        raise ContractError("power artifact has fewer than two rows")
    return rows


def integrate_power(rows: list[dict[str, Any]], start_us: int, end_us: int,
                    require_quality: bool) -> dict[str, Any]:
    if type(start_us) is not int or type(end_us) is not int or end_us <= start_us:
        raise ContractError("invalid paid window")
    if rows[0]["t_us"] > start_us or rows[-1]["t_us"] < end_us:
        raise ContractError("power rows do not bracket the paid window")
    limits = {row["power_limit_mw"] for row in rows if start_us <= row["t_us"] <= end_us}
    if len(limits) != 1:
        raise ContractError("power limit changed in the paid window")
    energy_nj = 0
    bracketed = [row for row in rows if start_us <= row["t_us"] <= end_us]
    before = [row for row in rows if row["t_us"] <= start_us]
    if not before:
        raise ContractError("missing left power bracket")
    previous_power = before[-1]["power_mw"]
    updates = 0
    for row in rows:
        if row["t_us"] <= start_us:
            continue
        if row["t_us"] >= end_us:
            break
        if row["power_mw"] != previous_power:
            updates += 1
        previous_power = row["power_mw"]
    sliced = before[-1:] + bracketed + [row for row in rows if row["t_us"] >= end_us][:1]
    deduped = []
    for row in sliced:
        if not deduped or row["t_us"] != deduped[-1]["t_us"]:
            deduped.append(row)
    max_gap_us = max(right["t_us"] - left["t_us"] for left, right in zip(deduped, deduped[1:]))
    for left, right in zip(rows, rows[1:]):
        lo = max(left["t_us"], start_us)
        hi = min(right["t_us"], end_us)
        if hi > lo:
            energy_nj += left["power_mw"] * (hi - lo)
    if energy_nj <= 0:
        raise ContractError("integrated energy is not positive")
    if require_quality and updates < MIN_UPDATES:
        raise ContractError(f"independent updates {updates} < {MIN_UPDATES}")
    if max_gap_us > MAX_GAP_US:
        raise ContractError(f"maximum sample gap {max_gap_us} > {MAX_GAP_US}")
    duration_us = end_us - start_us
    return {
        "energy_nj": energy_nj,
        "duration_us": duration_us,
        "avg_power_mw": energy_nj // duration_us,
        "independent_updates": updates,
        "max_sample_gap_us": max_gap_us,
        "power_limit_mw": next(iter(limits)),
        "uncertainty_nj": NVML_UNCERTAINTY_MW * duration_us,
    }


def validate_result_record(value: dict[str, Any], reference: list[int], label: str,
                           launch_id: int, session_end: str) -> None:
    expected = {
        "batch_size", "elapsed_us", "host_pid", "launch_id", "n_gen", "outcome",
        "request_count", "route_wall_us", "schema", "session_end", "token_ids",
    }
    if set(value) != expected or value.get("schema") != "layersplit-persistent-result-v1" \
            or value.get("outcome") != "completed" or value.get("batch_size") != BATCH \
            or value.get("request_count") != BATCH or value.get("n_gen") != N_GEN \
            or value.get("launch_id") != launch_id or value.get("session_end") != session_end:
        raise ContractError(f"{label} persistent result identity failed")
    for key in ("elapsed_us", "host_pid", "route_wall_us"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ContractError(f"{label} persistent result has invalid {key}")
    tokens = value.get("token_ids")
    if type(tokens) is not list or len(tokens) != BATCH \
            or any(item != reference for item in tokens):
        raise ContractError(f"{label} persistent tokens differ from the reference")


def summarize(rows: list[dict[str, Any]], require_power_quality: bool) -> dict[str, Any]:
    expected_schedule = FULL_SCHEDULE if require_power_quality else (("P0", 0), ("P2", 0))
    if [(row.get("label"), row.get("pair")) for row in rows] != list(expected_schedule):
        raise ContractError("row schedule differs from the frozen order")
    for row in rows:
        if row.get("gemma_batches") != (LOW_COHORTS if require_power_quality else 2) \
                or row.get("gemma_requests") != row["gemma_batches"] * BATCH \
                or row.get("gemma_tokens") != row["gemma_requests"] * N_GEN:
            raise ContractError("row has unmatched Gemma work")
        if row.get("all_tokens_match") is not True or row.get("all_low_inside_bge") is not True:
            raise ContractError("row correctness or overlap gate failed")
        low = row.get("low_route_wall_us")
        if type(low) is not list or len(low) != row["gemma_batches"] \
                or any(type(value) is not int or value <= 0 for value in low):
            raise ContractError("row has invalid low latency samples")
        if max(low) > LOW_SLO_US:
            raise ContractError("low-priority absolute SLO failed")
        bge = row.get("bge")
        if type(bge) is not dict or type(bge.get("encodes")) is not int \
                or bge["encodes"] <= 0 or type(bge.get("latency_samples_us")) is not list:
            raise ContractError("row has invalid BGE work")
    encodes = {row["bge"]["encodes"] for row in rows}
    if len(encodes) != 1:
        raise ContractError("BGE work differs across rows")
    by_label = {label: [row for row in rows if row["label"] == label] for label in ("P0", "P2")}
    high_p95 = {
        label: statistics.median(
            percentile([int(round(value)) for value in row["bge"]["latency_samples_us"]], 95, 100)
            for row in selected)
        for label, selected in by_label.items()
    }
    high_ratio = high_p95["P2"] / high_p95["P0"]
    summary = {
        "matched_bge_encodes_per_row": next(iter(encodes)),
        "matched_gemma_requests_per_row": rows[0]["gemma_requests"],
        "matched_gemma_tokens_per_row": rows[0]["gemma_tokens"],
        "high_p95_us_median": high_p95,
        "high_p95_ratio_p2_over_p0": high_ratio,
        "high_priority_gate": high_ratio <= HIGH_P95_RATIO_MAX,
        "low_priority_gate": True,
    }
    if not require_power_quality:
        summary["screen_pass"] = summary["high_priority_gate"]
        return summary
    pairs = []
    for pair in range(3):
        control = next(row for row in rows if row["label"] == "P0" and row["pair"] == pair)
        treatment = next(row for row in rows if row["label"] == "P2" and row["pair"] == pair)
        c_energy = control["power"]["energy_nj"]
        t_energy = treatment["power"]["energy_nj"]
        uncertainty = control["power"]["uncertainty_nj"] + treatment["power"]["uncertainty_nj"]
        pairs.append({
            "pair": pair,
            "saving_nj": c_energy - t_energy,
            "saving_fraction": 1.0 - t_energy / c_energy,
            "uncertainty_nj": uncertainty,
            "lower_bound_saving_nj": c_energy - t_energy - uncertainty,
        })
    median_saving = statistics.median(item["saving_fraction"] for item in pairs)
    median_lower = statistics.median(item["lower_bound_saving_nj"] for item in pairs)
    summary.update({
        "pairs": pairs,
        "selected_gpu_energy_saving_fraction_median": median_saving,
        "selected_gpu_lower_bound_saving_nj_median": median_lower,
        "selected_gpu_board_relief_gate": median_saving > 0 and median_lower > 0,
    })
    summary["overall_pass"] = summary["high_priority_gate"] \
        and summary["selected_gpu_board_relief_gate"]
    return summary

