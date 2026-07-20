#!/usr/bin/env python3
"""Pure contract checks for the S18 two-phone R1 acquisition."""

from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
S16_CONTRACT = HERE.parent / "s16_mixed_persistent_energy/experiment_contract.py"
_spec = importlib.util.spec_from_file_location("s16_experiment_contract", S16_CONTRACT)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load the S16 evidence utilities")
_s16 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _s16
_spec.loader.exec_module(_s16)

ContractError = _s16.ContractError
canonical = _s16.canonical
digest_file = _s16.digest_file
integrate_power = _s16.integrate_power
parse_power_jsonl = _s16.parse_power_jsonl
percentile = _s16.percentile
strict_object = _s16.strict_object


PHONE_BATCH = 32
CONTROL_BATCH = 32
BATCHES_PER_ROUND = 2
N_GEN = 8
SCREEN_ROUNDS = 2
FULL_ROUNDS = 6
OP15_SLO_US = 5_000_000
OP12_SLO_US = 12_000_000
HIGH_P95_RATIO_MAX = 1.05
RELIEF_FRACTION_MIN = 0.10
OP12_EXCHANGE_CREDIT = 2
SCREEN_SCHEDULE = (("P0", 0), ("P4", 0))
FULL_SCHEDULE = (("P0", 0), ("P4", 0), ("P4", 1),
                 ("P0", 1), ("P0", 2), ("P4", 2))


def validate_result_record(value: dict[str, Any], reference: list[int], batch: int,
                           launch_id: int, session_end: str) -> None:
    expected = {
        "batch_size", "elapsed_us", "host_pid", "launch_id", "n_gen", "outcome",
        "request_count", "route_wall_us", "schema", "session_end", "token_ids",
    }
    if set(value) != expected \
            or value.get("schema") != "layersplit-persistent-result-v1" \
            or value.get("outcome") != "completed" \
            or value.get("batch_size") != batch \
            or value.get("request_count") != batch \
            or value.get("n_gen") != N_GEN \
            or value.get("launch_id") != launch_id \
            or value.get("session_end") != session_end:
        raise ContractError("persistent result identity failed")
    for key in ("elapsed_us", "host_pid", "route_wall_us"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ContractError(f"persistent result has invalid {key}")
    tokens = value.get("token_ids")
    if type(tokens) is not list or len(tokens) != batch \
            or any(type(item) is not list or item != reference for item in tokens):
        raise ContractError("persistent tokens differ from the reference")


def _valid_window(value: object) -> bool:
    return type(value) is list and len(value) == 2 \
        and all(type(item) is int for item in value) and value[1] > value[0]


def _validate_row(row: dict[str, Any], label: str, pair: int, rounds: int) -> None:
    if row.get("label") != label or row.get("pair") != pair:
        raise ContractError("row order differs from the frozen schedule")
    if row.get("rounds") != rounds \
            or row.get("gemma_requests") != rounds * BATCHES_PER_ROUND * CONTROL_BATCH \
            or row.get("gemma_tokens") != rounds * BATCHES_PER_ROUND * CONTROL_BATCH * N_GEN \
            or row.get("all_tokens_match") is not True \
            or row.get("all_low_inside_bge") is not True:
        raise ContractError("row work or correctness gate failed")
    bge = row.get("bge")
    if type(bge) is not dict or type(bge.get("encodes")) is not int \
            or bge["encodes"] <= 0 or type(bge.get("latency_samples_us")) is not list \
            or not bge["latency_samples_us"]:
        raise ContractError("row has invalid BGE work")
    peak = row.get("selected_gpu_peak_memory_mib")
    if type(peak) is not int or peak <= 0:
        raise ContractError("row has no selected-GPU memory observation")
    if label == "P0":
        walls = row.get("control_route_wall_us")
        windows = row.get("control_windows_us")
        if type(walls) is not list or len(walls) != rounds * BATCHES_PER_ROUND \
                or any(type(value) is not int or value <= 0 for value in walls) \
                or type(windows) is not list or len(windows) != rounds * BATCHES_PER_ROUND \
                or any(not _valid_window(window) for window in windows):
            raise ContractError("P0 low-work evidence is invalid")
    else:
        op12_count = min(rounds, OP12_EXCHANGE_CREDIT)
        op15_count = rounds * BATCHES_PER_ROUND - op12_count
        walls = row.get("phone_route_wall_us")
        windows = row.get("phone_windows_us")
        if type(walls) is not dict or set(walls) != {"op15", "op12"} \
                or type(walls["op15"]) is not list or len(walls["op15"]) != op15_count \
                or type(walls["op12"]) is not list or len(walls["op12"]) != op12_count \
                or any(type(value) is not int or value <= 0
                       for values in walls.values() for value in values):
            raise ContractError("P4 route latency evidence is invalid")
        if max(walls["op15"]) > OP15_SLO_US or max(walls["op12"]) > OP12_SLO_US:
            raise ContractError("phone route SLO failed")
        if type(windows) is not dict or set(windows) != {"op15", "op12"} \
                or type(windows["op15"]) is not list or len(windows["op15"]) != op15_count \
                or type(windows["op12"]) is not list or len(windows["op12"]) != op12_count \
                or any(not _valid_window(window)
                       for values in windows.values() for window in values):
            raise ContractError("P4 route windows are invalid")
        overlap = row.get("round_overlap_us")
        if type(overlap) is not list or len(overlap) != op12_count \
                or any(type(value) is not int or value <= 0 for value in overlap):
            raise ContractError("phone routes did not overlap")
        assignments = row.get("route_assignments")
        expected_assignments = [
            ["op15", "op12" if index < op12_count else "op15"]
            for index in range(rounds)
        ]
        if assignments != expected_assignments:
            raise ContractError("P4 route assignment differs from the credit policy")
        completion = row.get("group_completion_us")
        if type(completion) is not list or len(completion) != rounds \
                or any(type(values) is not list or len(values) != 2
                       or any(type(value) is not int or value <= 0 for value in values)
                       for values in completion):
            raise ContractError("P4 group completion evidence is invalid")
        if any(values[0] > OP15_SLO_US or values[1] > OP12_SLO_US
               for values in completion):
            raise ContractError("P4 class SLO including queueing failed")


def summarize(rows: list[dict[str, Any]], full: bool) -> dict[str, Any]:
    schedule = FULL_SCHEDULE if full else SCREEN_SCHEDULE
    rounds = FULL_ROUNDS if full else SCREEN_ROUNDS
    if len(rows) != len(schedule):
        raise ContractError("row count differs from the frozen schedule")
    for row, (label, pair) in zip(rows, schedule):
        _validate_row(row, label, pair, rounds)
    encodes = {row["bge"]["encodes"] for row in rows}
    if len(encodes) != 1:
        raise ContractError("BGE work differs across rows")
    by_label = {
        label: [row for row in rows if row["label"] == label]
        for label in ("P0", "P4")
    }
    high_p95 = {
        label: statistics.median(
            percentile([int(round(value)) for value in row["bge"]["latency_samples_us"]],
                       95, 100)
            for row in selected
        )
        for label, selected in by_label.items()
    }
    ratio = high_p95["P4"] / high_p95["P0"]
    summary: dict[str, Any] = {
        "matched_bge_encodes_per_row": next(iter(encodes)),
        "matched_gemma_requests_per_row": rounds * BATCHES_PER_ROUND * CONTROL_BATCH,
        "matched_gemma_tokens_per_row": rounds * BATCHES_PER_ROUND * CONTROL_BATCH * N_GEN,
        "high_p95_us_median": high_p95,
        "high_p95_ratio_p4_over_p0": ratio,
        "high_priority_gate": ratio <= HIGH_P95_RATIO_MAX,
        "low_correctness_slo_overlap_gate": True,
        "selected_gpu_peak_memory_mib_max": max(
            row["selected_gpu_peak_memory_mib"] for row in rows),
    }
    if not full:
        summary["screen_pass"] = summary["high_priority_gate"]
        return summary
    pairs = []
    for pair in range(3):
        control = next(row for row in rows if row["label"] == "P0" and row["pair"] == pair)
        treatment = next(row for row in rows if row["label"] == "P4" and row["pair"] == pair)
        c_energy = control["power"]["energy_nj"]
        t_energy = treatment["power"]["energy_nj"]
        uncertainty = control["power"]["uncertainty_nj"] \
            + treatment["power"]["uncertainty_nj"]
        pairs.append({
            "pair": pair,
            "saving_nj": c_energy - t_energy,
            "saving_fraction": 1.0 - t_energy / c_energy,
            "uncertainty_nj": uncertainty,
            "lower_bound_saving_nj": c_energy - t_energy - uncertainty,
        })
    median_saving = statistics.median(item["saving_fraction"] for item in pairs)
    median_lower = statistics.median(item["lower_bound_saving_nj"] for item in pairs)
    energy_gate = median_saving >= RELIEF_FRACTION_MIN and median_lower > 0
    summary.update({
        "pairs": pairs,
        "selected_gpu_energy_saving_fraction_median": median_saving,
        "selected_gpu_lower_bound_saving_nj_median": median_lower,
        "selected_gpu_board_relief_gate": energy_gate,
        "overall_pass": summary["high_priority_gate"] and energy_gate,
    })
    return summary
